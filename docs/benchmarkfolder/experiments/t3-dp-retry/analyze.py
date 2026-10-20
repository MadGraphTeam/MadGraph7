#!/usr/bin/env python3
"""Validate explicit-retry output and compare every retained physical component."""
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import ninjars_benchmark as bench
from madgraph.various.banner import MadLoopParam

spec = importlib.util.spec_from_file_location(
    "unconditional_comparison", HERE.parent / "t3-compensation-performance/run.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def rows(path):
    result = []
    for line in path.read_text().splitlines():
        fields = line.split()
        assert len(fields) == 8
        row = {"point": int(fields[0]), "order": int(fields[1]),
               "status": int(fields[2]),
               "coefficients": list(map(float, fields[3:7])),
               "accuracy": float(fields[7])}
        bench.validate_status(row["status"], 8)
        assert all(math.isfinite(x) for x in row["coefficients"] + [row["accuracy"]])
        assert row["accuracy"] >= 0
        result.append(row)
    return result


def main(case, label):
    directory = HERE / case / "runs" / label
    values = rows(directory / "values.dat")
    mapping = directory / "point-map.json"
    indices = json.loads(mapping.read_text()) if mapping.exists() else list(
        range(1, int((directory/"points.dat").read_text().splitlines()[0])+1))
    threshold = float(MadLoopParam(str(directory/"MadLoopParams.dat"))["MLStabThres"])
    allocated, orders, born_orders = map(
        int, (directory / "dimensions.dat").read_text().split())
    assert allocated == max(orders, born_orders) and born_orders <= orders
    assert [(r["point"], r["order"]) for r in values] == [
        (p, o) for p in range(1, len(indices)+1) for o in range(orders+1)]
    retry = []
    for line in (directory / "retry-stats.dat").read_text().splitlines():
        words = line.split()
        assert len(words) == 10
        point, status, count, passed, failed, hard = map(int, words[:6])
        native_acc, compensated_acc = map(float, words[8:])
        assert words[6] == "F" and words[7] in ("T", "F")
        assert count in (0, 1) and passed+failed == count and hard in (0, 1)
        assert (words[7] == "T") == bool(count)
        assert not hard or count == 0
        assert math.isfinite(native_acc+compensated_acc)
        assert [r["status"] for r in values if r["point"] == point] == [status]*(orders+1)
        if count:
            assert native_acc >= threshold
            assert (compensated_acc < threshold) == bool(passed)
            assert status == (218 if passed else 322)
        retry.append({"point": indices[point-1], "status": status,
                      "retry_count": count, "passed": passed, "failed": failed,
                      "hard_skip": hard, "native_accuracy": native_acc,
                      "compensated_accuracy": compensated_acc})
    assert len(retry) == len(indices)
    counts = Counter(str(row["status"]) for row in retry)
    timed = dict(map(lambda line: tuple(map(int, line.split())),
                     (directory / "timed_status.dat").read_text().splitlines()))
    metadata = json.loads((directory / "run.json").read_text())
    loops = int(metadata["command"][-1])
    assert timed == {int(status): count*loops for status, count in counts.items()}
    for row in values:
        row["point"] = indices[row["point"]-1]
    references = HERE.parent / "t3-compensation-performance/first-three-cases-20261020" / case / "exploratory"
    comparisons = {}
    sources = [(variant, references/f"{variant}-0-values.dat")
               for variant in ("native", "guarded")] if not case.startswith("control-") else [
        (path.parent.parent.name+"/"+path.name.removesuffix("-fresh.json"), path)
        for path in (ROOT/"docs/benchmarkfolder/policy-v2/representative-runs").glob(
            f"*/data/{case}-*-fresh.json")]
    assert sources, "no retained physical reference"
    ordinary_identity = None
    for variant, path in sources:
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            assert data["points_sha256"] == metadata["points_sha256"]
            assert data["parameter_card_sha256"] == metadata["parameter_card_sha256"]
            reference = data["rows"]
        else:
            reference = rows(path)
        reference = [row for row in reference if row["point"] in indices]
        by_key = {(row["point"], row["order"]): row for row in reference}
        reference = [by_key[(row["point"], row["order"])] for row in values]
        comparisons[variant] = comparison.agreement(values, reference)
        comparisons[variant]["source"] = str(path.relative_to(ROOT))
        comparisons[variant]["source_sha256"] = comparison.sha(path)
        comparisons[variant]["required"] = "representative-original-controls" not in str(path)
        comparisons[variant]["role"] = (
            "literal-original diagnostic" if not comparisons[variant]["required"]
            else "corrected-cohort physical reference")
        if variant == "native" and (directory/"MadLoopParams.dat").read_bytes() == (
                references.parent/"native-MadLoopParams.dat").read_bytes():
            retried = {row["point"] for row in retry if row["retry_count"] or row["hard_skip"]}
            ordinary = [(a, b) for a, b in zip(values, reference) if a["point"] not in retried]
            assert all(a == b for a, b in ordinary), "ordinary DP path changed"
            ordinary_identity = {
                "points": sum(row["point"] not in retried for row in retry), "rows": len(ordinary),
                "all_printed_coefficients_accuracy_and_status_exact": True}
    result = {"passed": all(c["passed"] for c in comparisons.values() if c["required"]),
              "acceptance_eligible": False, "status_counts": dict(counts),
              "retry_points": [row for row in retry if row["retry_count"] or row["hard_skip"]],
              "all_points": retry, "physical_comparisons": comparisons}
    if ordinary_identity is not None:
        result["ordinary_path_identity"] = ordinary_identity
    (directory / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "all_points"}, indent=2))
    assert result["passed"], "physical coefficient disagreement"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="qcd-3")
    parser.add_argument("--label", default="full-100")
    args = parser.parse_args()
    main(args.case, args.label)
