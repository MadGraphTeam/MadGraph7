#!/usr/bin/env python3
"""Summarize a campaign's ordinary numerical lanes (11 cases x 6 lanes).

Reads REPORT/experiments/numerical/<label>/results.json for every case of the
campaign and writes a compact JSON, a CSV and a Markdown table: 3e-6 pass,
regression failures, 1e-8 target misses (the same LoopMEComparator decision as
tirust_benchmark.compare_rows), the largest symmetric finite residual, accepted
DP/QP counts and the cold start time. With --predecessor the DP/QP counts and
target misses of an earlier campaign are shown alongside.

  tirust_numerical_summary.py --campaign tirust-9a781998-1 [--predecessor entry-latch-cache-guard-1]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = ROOT / "docs/benchmarkfolder/campaigns"
LANES = (("baseline", 1), ("candidate", 1), ("candidate", 6), ("candidate", 7), ("candidate", 8), ("candidate", 9))
COLUMNS = ("born", "finite", "single", "double")


def accepted(residual, threshold):
    symmetric, pole = residual["symmetric_relative_error"], residual["pole_to_finite"]
    return (symmetric is not None and symmetric < threshold) or (pole is not None and pole < threshold)


def lane_summary(result, target=1e-8):
    item = {"status": result["status"]}
    if result["status"] != "complete":
        item["error"] = result.get("error")
        return item
    observation = result["observation"]
    counts = observation.get("backend_precision_counts", {})
    item.update(dp=sum(v for k, v in counts.items() if k.endswith(":DP")),
                qp=sum(v for k, v in counts.items() if k.endswith(":QP")),
                status_counts=observation.get("status_counts"), cold_seconds=observation.get("cold_seconds"))
    comparison = result.get("comparison")
    if comparison is None:
        item["reference"] = True
        return item
    misses = [r for r in comparison["residuals"] if not accepted(r, target)]
    finite = [r["symmetric_relative_error"] for r in comparison["residuals"]
              if r["column"] == 1 and r["symmetric_relative_error"] is not None]
    item.update(passed=comparison["passed"], failures=len(comparison["failures"]),
                target_misses=len(misses),
                target_misses_by_column={c: sum(1 for r in misses if r["column"] == i) for i, c in enumerate(COLUMNS)},
                target_miss_points=sorted({r["point"] for r in misses}),
                max_symmetric_finite=max(finite) if finite else None,
                max_relative_error=comparison["maximum_relative_error"])
    return item


def find_results(campaign, case):
    numerical = CAMPAIGNS / campaign / "experiments/numerical"
    for label in (f"ordinary-{case}", f"full100-warmup-{case}-1"):
        path = numerical / label / "results.json"
        if path.is_file():
            return path
    return None


def collect(campaign):
    cases = json.loads((CAMPAIGNS / campaign / "cases.json").read_text())["cases"]
    summary = {}
    for case in cases:
        path = find_results(campaign, case["id"])
        lanes = {}
        if path is not None:
            for result in json.loads(path.read_text())["results"]:
                lanes[f"{result['source']}-{result['backend']}"] = lane_summary(result)
        summary[case["id"]] = {"process": case["process"], "model": case["model"], "role": case["role"],
                               "results": str(path.relative_to(ROOT)) if path else None, "lanes": lanes}
    return summary


def table(summary, predecessor=None):
    lines = ["| Case | Lane | 3e-6 | 1e-8 misses (points) | max sym. finite | DP/QP | previous DP/QP, misses |",
             "|---|---|---|---|---|---|---|"]
    for case, entry in summary.items():
        for source, backend in LANES:
            key = f"{source}-{backend}"
            lane = entry["lanes"].get(key)
            if lane is None:
                lines.append(f"| {case} | {key} | pending | | | | |")
                continue
            if lane["status"] != "complete":
                lines.append(f"| {case} | {key} | **{lane['status']}** | | | | |")
                continue
            old = ""
            if predecessor and key in predecessor.get(case, {}).get("lanes", {}):
                p = predecessor[case]["lanes"][key]
                if p["status"] == "complete":
                    old = f"{p['dp']}/{p['qp']}" + (f", {p['target_misses']}" if "target_misses" in p else "")
            if lane.get("reference"):
                lines.append(f"| {case} | {key} | reference | | | {lane['dp']}/{lane['qp']} | {old} |")
                continue
            finite = lane["max_symmetric_finite"]
            lines.append(f"| {case} | {key} | {'pass' if lane['passed'] else '**FAIL**'} | "
                         f"{lane['target_misses']} ({len(lane['target_miss_points'])}) | "
                         f"{finite:.2e} | {lane['dp']}/{lane['qp']} | {old} |")
    return "\n".join(lines) + "\n"


def rows(summary):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["case", "lane", "status", "passed_3e-6", "failures", "target_misses_1e-8",
                     "target_miss_points", "max_symmetric_finite", "dp", "qp", "cold_seconds"])
    for case, entry in summary.items():
        for source, backend in LANES:
            lane = entry["lanes"].get(f"{source}-{backend}", {"status": "pending"})
            writer.writerow([case, f"{source}-{backend}", lane["status"], lane.get("passed", ""),
                             lane.get("failures", ""), lane.get("target_misses", ""),
                             len(lane.get("target_miss_points", [])) if "target_miss_points" in lane else "",
                             lane.get("max_symmetric_finite", ""), lane.get("dp", ""), lane.get("qp", ""),
                             lane.get("cold_seconds", "")])
    return out.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--predecessor")
    parser.add_argument("--out", type=Path, help="directory for summary.json/.csv/.md (default: print only)")
    args = parser.parse_args(argv)
    summary = collect(args.campaign)
    predecessor = collect(args.predecessor) if args.predecessor else None
    complete = [lane for entry in summary.values() for lane in entry["lanes"].values()]
    totals = {"lanes_complete": sum(lane["status"] == "complete" for lane in complete),
              "lanes_expected": len(summary) * len(LANES),
              "lanes_failed_3e-6": sum(1 for lane in complete if lane.get("passed") is False),
              "lanes_meeting_1e-8": sum(1 for lane in complete if lane.get("target_misses") == 0)}
    document = {"campaign": args.campaign, "predecessor": args.predecessor, "totals": totals, "cases": summary}
    markdown = table(summary, predecessor)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "numerical-summary.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        (args.out / "numerical-summary.csv").write_text(rows(summary))
        (args.out / "numerical-summary.md").write_text(markdown)
    print(json.dumps(totals))
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
