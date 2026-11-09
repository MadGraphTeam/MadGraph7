#!/usr/bin/env python3
"""Recheck retained tensor results without executing any numerical library."""
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "analysis-protocol"))
from tirust_benchmark_poles import parse_results, pole_split_checks

summaries = []
for label in ("final-qcd-n1-poles-r2", "final-qcd-n4-poles-r4"):
    directory = HERE / label
    selected = json.loads((directory / "specification.json").read_text())["selected"]
    text = (directory / "pole-values.dat").read_text()
    pairs = parse_results(text, selected)
    splits = pole_split_checks(text, selected)
    unavailable = [{"fixture": item["fixture"]["name"], "ctmode": item["fixture"]["ctmode"],
                    "pole_mode": item["fixture"]["pole_mode"]}
                   for item in pairs if not item["stable"]]
    summary = {"label": label, "comparisons": len(pairs),
        "status_counts": dict(Counter(item["status"] for item in pairs)),
        "precision_pass_counts": {str(precision): sum(item["status"] == "passed" and
            item["fixture"]["ctmode"] == precision for item in pairs) for precision in (1, 4)},
        "unavailable": unavailable,
        "all_served_pairs_pass": all(item["status"] == "passed" for item in pairs if item["stable"]),
        "complete": all(item["status"] == "passed" for item in pairs),
        "maximum_served_scaled_residual": str(max(Decimal(row["scaled_residual"])
            for item in pairs for row in item["residuals"])),
        "pole_split_status_counts": dict(Counter(item["status"] for item in splits)),
        "pole_split_checks": splits,
        "raw_sha256": hashlib.sha256((directory / "pole-values.dat").read_bytes()).hexdigest()}
    (directory / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    summaries.append(summary)
summary = {"scope": "Generated tensor adapter correctness diagnostics; no performance acceptance",
           "runs": summaries, "compared_pairs": sum(run["comparisons"] for run in summaries),
           "passed_pairs": sum(run["status_counts"].get("passed", 0) for run in summaries),
           "unavailable_pairs": sum(len(run["unavailable"]) for run in summaries)}
cache_path = HERE / "final-qcd-n1-cache/results.json"
if cache_path.is_file():
    cache = json.loads(cache_path.read_text())
    summary["cache"] = {"passed": cache["passed"], "records": len(cache["results"]),
        "maximum_scaled_residual": str(max(Decimal(item["maximum_scaled_residual"])
            for item in cache["results"])),
        "results_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest()}
(HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({key: value for key, value in summary.items() if key != "runs"}, indent=2))
