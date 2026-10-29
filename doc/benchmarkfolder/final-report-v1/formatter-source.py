"""Offline display view of an already validated CLI-consumer snapshot.

This formatter does not validate or accept physics. Run the concrete public CLI
consumer first, then bind its materialized JSON by SHA256. Raw statistics remain
unchanged; optional zero-weight polarized comparisons are omitted from display.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

ORDER = ("qcd-1", "ew-1", "qcd-2", "ew-2", "qcd-3", "ew-3", "qcd-4", "ew-4",
         "control-gg-h", "control-gg-hh", "profile-gg-hhh", "profile-gg-hhhh")
PAGES = (("SM: one or two final-state particles", ORDER[:4]),
         ("SM: three or four final-state particles", ORDER[4:8]),
         ("Loop-induced amplitudes: H, HH, HHH and HHHH", ORDER[8:]))


def format_snapshot(snapshot):
    if snapshot.get("schema") != "ninjars-cli-profile-report-v1" or snapshot.get("requested_tools") != [6, 8]:
        raise ValueError("Use the materialized two-tool CLI-consumer snapshot")
    if set(snapshot["cases"]) != set(ORDER):
        raise ValueError("The final CLI presentation retains all twelve declared processes, including both 2-to-4 cases")
    rows, optional, suppressed = {}, [], []
    for cid in ORDER:
        case = snapshot["cases"][cid]
        complete = case["status"] == "complete_comparable"
        observations = case["observations"]
        stage_counts = {
            backend: {name: [record["stage_point_counts"][name] for record in records]
                      for name in ("unpolarized_full", "unpolarized_skip", "polarized_full", "polarized_skip")}
            for backend, records in observations.items()}
        row = {"id": cid, "process": case["process"], "status": case["status"],
               "sessions": case["sessions"], "required_sessions_per_backend": case["required_sessions_per_backend"],
               "actual_stage_counts": stage_counts,
               "quantities": copy.deepcopy(case["timings"]["helicity_sum"]),
               "primary_mode": "full-helicity-color-sum",
               "failed_attempts": copy.deepcopy(case["failed_attempts"])}
        row["stage_counts"] = "; ".join(
            name + " FULL/SKIP " + str(stage_counts[backend]["unpolarized_full"]) + "/" +
            str(stage_counts[backend]["unpolarized_skip"])
            for backend, name in (("6", "Ninja"), ("8", "NRS")))
        # Do not turn absent/incomparable sessions into a complete final table.
        # The source consumer still retains partial actual statistics and errors.
        rows[cid] = row
        weights = {backend: [record["selected_helicity_weight"] for record in records]
                   for backend, records in observations.items()}
        positive = complete and all(type(w) in (int, float) and math.isfinite(w) and w > 0
                                    for series in weights.values() for w in series)
        if positive:
            optional.append({"id": cid, "weights": weights,
                             "quantities": copy.deepcopy(case["timings"]["single_helicity"]),
                             "actual_stage_counts": stage_counts,
                             "scope": "Optional positive-weight selected-helicity diagnostic; not the full-sum primary comparison"})
        else:
            suppressed.append({"id": cid, "weights": weights,
                               "display": "grey; no optional polarized performance ratio",
                               "reason": "Zero/nonpositive or unavailable saved weight, or incomplete/incomparable sessions",
                               "raw_diagnostic_preserved": True})
    return {"pages": [{"index": i, "title": title, "rows": [rows[cid] for cid in ids]}
                       for i, (title, ids) in enumerate(PAGES)],
            "all_twelve_sessions_complete_comparable": snapshot["complete"],
            "declared_scope": copy.deepcopy(snapshot["declared_scope"]),
            "runtime_pins": copy.deepcopy(snapshot["runtime_pins"]),
            "optional_positive_weight_polarized": optional,
            "suppressed_optional_polarized": suppressed,
            "retained_failed_attempts": snapshot["retained_failed_attempts"],
            "attempt_inventory_scope": snapshot["attempt_inventory_scope"],
            "timed_status_counts_available": False,
            "source_snapshot_preserved": True,
            "scope": "Display-only full-sum RED/SKIP/TOTAL view; actual consumer statistics copied unchanged"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--sha256", required=True, help="Hash of the actual, concrete-consumer materialization")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.sha256:
        raise ValueError("Materialized CLI snapshot changed")
    if args.output.exists():
        raise ValueError("Preserve earlier display views")
    view = format_snapshot(json.loads(raw))
    view["materialized_snapshot"] = {"path": str(args.snapshot), "sha256": args.sha256}
    view["formatter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(view, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
