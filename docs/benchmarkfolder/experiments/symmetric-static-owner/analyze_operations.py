#!/usr/bin/env python3
"""Bind and summarize the complete diagnostic operation trace."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from count_products import FIELDS

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    folder = HERE / args.run_id
    result = json.loads((folder / "result.json").read_text())
    assert result["complete"] and result["all_printed_rows_and_statuses_exact"]
    assert set(result["measurements"]) == {"original", "folded-counts"}
    assert all(len(v) == 1 for v in result["measurements"].values())
    original = folder / "original-0-values.dat"
    candidate = folder / "folded-counts-0-values.dat"
    assert original.read_bytes() == candidate.read_bytes()
    original_status = folder / "original-0-timed_status.dat"
    candidate_status = folder / "folded-counts-0-timed_status.dat"
    assert original_status.read_bytes() == candidate_status.read_bytes()
    proof = json.loads((folder / "folded-counts-proof.json").read_text())
    assert proof["operation_counter_fields"] == list(FIELDS)
    ops_path = folder / "folded-counts-sharing_operations.dat"
    calls_path = folder / "folded-counts-sharing_trace.dat"
    operations = [list(map(int, line.split())) for line in ops_path.read_text().splitlines()]
    calls = [list(map(int, line.split())) for line in calls_path.read_text().splitlines()]
    assert len(operations) == len(calls) == len(proof["helpers"])
    summaries = []
    totals = [0] * len(FIELDS)
    ranks = defaultdict(lambda: [0] * len(FIELDS))
    for helper, row, call in zip(proof["helpers"], operations, calls):
        assert len(row) == len(FIELDS) + 1 and len(call) == 3
        assert helper["index"] == row[0] == call[0]
        assert all(x >= 0 for x in row[1:] + call[1:])
        assert call[2] == 0, "Fallback hides a helper's original operation counts"
        data = dict(zip(FIELDS, row[1:]))
        assert row[1] == row[2] + row[3] + row[4]
        assert row[1] == row[8] + row[9] + row[10]
        assert row[7] == call[1] * 4 * 5 * 4
        for i, value in enumerate(row[1:]):
            totals[i] += value
            ranks[("terminal" if helper["terminal"] else "full")
                  + "-rank" + str(helper["rank"])][i] += value
        summaries.append(dict(**helper, fast_calls=call[1], fallback_calls=call[2],
            counts=data,
            reused_fraction=row[4] / row[1] if row[1] else 0.0))
    counts = dict(zip(FIELDS, totals))
    original_products, unshared, owners, reused, a_skip, b_skip = totals[:6]
    summary = dict(complete=True, acceptance_eligible=False,
        all_printed_values_accuracy_and_statuses_exact=True,
        helpers=summaries, counts=counts,
        counts_by_helper_kind_rank={k: dict(zip(FIELDS, v)) for k, v in ranks.items()},
        saved_product_fraction_after_zero_skips=reused / original_products,
        A_zero_skip_fraction_after_B_zero_skip=a_skip / (original_products + a_skip),
        B_zero_skip_fraction_of_possible_slots=b_skip / (original_products + a_skip + b_skip),
        pure_real_or_imaginary_B_fraction_of_eligible_products=(totals[7] + totals[8]) / original_products,
        fast_calls=sum(x["fast_calls"] for x in summaries),
        fallback_calls=sum(x["fallback_calls"] for x in summaries),
        evidence_sha256={p.name: sha(p) for p in (
            folder / "result.json", folder / "folded-counts-proof.json",
            original, candidate, original_status, candidate_status, ops_path, calls_path)},
        analyzer_sha256=sha(Path(__file__)),
        scope="One full unchanged 100-point QCD3 original-Ninja6 evaluation pass. "
              "Counters cover the selected private QP helper calls only. "
              "Original products are the operations that the original updater "
              "would execute after its existing A/B zero skips; each equals "
              "unshared+owner+reused. B skips count entire potential I/L slots "
              "before checking A, so there is no overlap with A skips. Phase "
              "counts are weighted by original eligible products, not helper "
              "calls. Instrumented runtime is not performance evidence; exact "
              "printed outputs do not expose every internal QP intermediate.")
    path = folder / "operation-summary.json"
    assert not path.exists()
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "counts", "saved_product_fraction_after_zero_skips",
        "A_zero_skip_fraction_after_B_zero_skip",
        "B_zero_skip_fraction_of_possible_slots",
        "pure_real_or_imaginary_B_fraction_of_eligible_products",
        "fast_calls", "fallback_calls")}, indent=2))


if __name__ == "__main__":
    main()
