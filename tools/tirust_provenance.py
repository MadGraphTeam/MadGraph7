#!/usr/bin/env python3
"""Per-point TIRust arithmetic provenance on a campaign's ordinary points.

Runs the stress driver (tools/tirust_stress_driver.f) on a private clone of the
campaign's candidate export with backend 9 alone, the campaign's natural policy
(CTModeRun=-1, MLStabThres and caches from cases.json), and bridge route hooks
around every amplitude call. Each of the 100 frozen points is classified as:

  dp_native  accepted DP (status 219); only Direct/MethodChoice binary64 reductions
  dp_dd      accepted DP; at least one tensor lane needed the double-double tier
  dp_mpfr    accepted DP; at least one tensor lane needed 256/512-bit MPFR
  qp         genuine binary128-input QP rescue (status 323)
  failed     refused, ambiguous routes or failure latch

MadLoop clears the TIRust numerical caches and its Fortran coefficient cache at
each new phase-space point, so a point's route deltas cover every reduction its
amplitude reused. Accepted statuses are cross-checked against the campaign's
numerical lane for the same points.

  run --case C --cpu N [--variant V]   one case; V = default | cache-off | small-budget
  summarize                            <report>/provenance/provenance-summary.{json,md}

The cache-off (TIRustGlobalCache=0) and small-budget (TIRustCacheMaxBytes=1 MiB,
forcing insertion refusals) variants must reproduce every coefficient, accuracy
and status of the default run bit for bit: each cache entry is a pure function
of its exact key.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as bench  # noqa: E402
import tirust_benchmark_cohorts as cohort  # noqa: E402
import tirust_stability as stability  # noqa: E402
from tirust_benchmark_worker import validate_status  # noqa: E402

CLASSES = ("dp_native", "dp_dd", "dp_mpfr", "qp", "failed")
VARIANTS = {"default": {}, "cache-off": {"TIRustGlobalCache": 0}, "small-budget": {"TIRustCacheMaxBytes": 1 << 20}}


def variant_dir(variant):
    return "provenance" if variant == "default" else f"provenance-{variant}"


def run(campaign, case_id, cpu, timeout=14400., variant="default"):
    config = stability.activate(campaign)
    case = next(c for c in config["cases"] if c["id"] == case_id)
    points = bench.REPORT / "points" / f"{case_id}.dat"
    report = bench.REPORT / "experiments" / variant_dir(variant) / case_id
    workspace = bench.WORK / variant_dir(variant) / case_id
    options = json.loads(json.dumps(config))
    options["candidate_parameters"].update(VARIANTS[variant])
    record = stability.run_driver(campaign, config, case, "candidate-9", cpu, points, report, workspace,
                                  options, timeout=timeout)
    print(json.dumps({"case": case_id, "status": record["status"]}), flush=True)
    return record


def classify(rows, route):
    if not rows or any(not r["finite"] or r["fpe_dp"] or r["fpe_qp"] for r in rows):
        return "failed"
    try:
        decoded = validate_status(rows[0]["status"], [9])
    except ValueError:
        return "failed"
    if decoded["precision"] == "QP":
        return "qp"
    if route is None:
        return "failed"
    return cohort.route_category("dp", route)


def lane_statuses(case_id):
    path = bench.REPORT / "experiments/numerical" / f"ordinary-{case_id}" / "candidate-9" / "result.json"
    if not path.is_file():
        return None
    rows = json.loads(path.read_text())["observation"]["rows"]
    return {row["point"]: row["status"] for row in rows if row["order"] == 0}


def summarize(campaign):
    stability.activate(campaign)
    base = bench.REPORT / "experiments/provenance"
    cases = {}
    for result_path in sorted(base.glob("*/result.json")):
        result = json.loads(result_path.read_text())
        case_id = result_path.parent.name
        entry = {"status": result["status"]}
        if result["status"] == "complete":
            routes = result.get("routes", {})
            points = sorted({r["point"] for r in result["rows"]})
            classes, totals = {}, [0] * 20
            for point in points:
                route = routes.get(str(point)) or routes.get(point)
                classes[point] = classify([r for r in result["rows"] if r["point"] == point], route)
                if route:
                    totals = [a + b for a, b in zip(totals, route["delta"])]
            entry.update(points=len(points), counts={c: sum(v == c for v in classes.values()) for c in CLASSES},
                         route_totals={"primary": totals[:10], "check": totals[10:]},
                         failed_points=[p for p, c in classes.items() if c == "failed"],
                         qp_points=[p for p, c in classes.items() if c == "qp"])
            for variant in VARIANTS:
                if variant == "default":
                    continue
                other = base.parent / variant_dir(variant) / case_id / "result.json"
                if other.is_file():
                    other = json.loads(other.read_text())
                    same = other["status"] == "complete" and [
                        (r["point"], r["order"], r["status"], r["coefficients"], r["accuracy"]) for r in other["rows"]] == [
                        (r["point"], r["order"], r["status"], r["coefficients"], r["accuracy"]) for r in result["rows"]]
                    entry[f"{variant}_identical"] = same
            lane = lane_statuses(case_id)
            if lane is not None:
                mine = {r["point"]: r["status"] for r in result["rows"] if r["order"] == 0}
                entry["numerical_lane_status_agreement"] = sum(lane.get(p) == s for p, s in mine.items())
                entry["numerical_lane_qp"] = sum(1 for s in lane.values() if s // 100 == 3)
        cases[case_id] = entry
    summary = {"campaign": campaign, "classes": CLASSES,
               "route_counter_order": ["Direct", "MethodChoice", "Tier106", "Tier256", "Tier512", "Refused",
                                       "unreported", "attempted106", "attempted256", "attempted512"],
               "cases": cases}
    out = bench.REPORT / "provenance"
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = ["| Case | binary64 only | DD tier | MPFR tier | genuine QP | failed | status = numerical lane | "
             "cache off / 1 MiB budget bit-identical |", "|---|---|---|---|---|---|---|---|"]
    for case_id, entry in cases.items():
        if entry["status"] != "complete":
            lines.append(f"| {case_id} | **{entry['status']}** | | | | | |")
            continue
        c = entry["counts"]
        agree = entry.get("numerical_lane_status_agreement")
        agreement = "" if agree is None else f"{agree}/{entry['points']}"
        caches = " / ".join({True: "yes", False: "**NO**", None: "pending"}[entry.get(f"{v}_identical")]
                            for v in ("cache-off", "small-budget"))
        lines.append(f"| {case_id} | {c['dp_native']} | {c['dp_dd']} | {c['dp_mpfr']} | {c['qp']} | {c['failed']} | "
                     f"{agreement} | {caches} |")
    (out / "provenance-summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "summarize"))
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--case")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--timeout", type=float, default=14400.)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="default")
    args = parser.parse_args(argv)
    if Path.cwd() != bench.ROOT:
        parser.error(f"Run from {bench.ROOT}")
    if args.command == "run":
        if not args.case or args.cpu is None:
            parser.error("run needs --case and --cpu")
        return 0 if run(args.campaign, args.case, args.cpu, args.timeout, args.variant)["status"] == "complete" else 1
    summarize(args.campaign)
    return 0


if __name__ == "__main__":
    sys.exit(main())
