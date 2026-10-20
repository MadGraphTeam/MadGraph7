#!/usr/bin/env python3
"""Rehash completed dumps/results and report independent oracle coverage."""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--expected-streams", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.result.exists():
        raise RuntimeError("Refuse to overwrite an aggregate result")
    run = json.loads((args.campaign/"run.json").read_text())
    assert run["complete"] and len(run["results"]) == args.expected_streams
    assert len({r["key"] for r in run["results"]}) == args.expected_streams
    assert run["factor_sha256"] == sha(args.campaign/"factor.py")
    assert run["oracle_driver_sha256"] == sha(args.campaign/"oracle-driver.py")
    assert run["runner_sha256"] == sha(args.campaign/"run_oracle.py")
    checked = []
    for record in run["results"]:
        work = args.campaign/record["key"]
        result_path = work/"exact-oracle.json"
        result = json.loads(result_path.read_text())
        assert record["returncode"] == 0 and result["complete"] and result["passed"]
        assert result["vertex"] == record["vertex"]
        assert result["oracle_sha256"] == sha(HERE/"oracle.py")
        assert result["proof_sha256"] == run["proof_sha256"]
        assert result["dump_sha256"] == record["gzip_sha256"] == sha(work/"dump.txt.gz")
        assert len(result["cases"]) == 10
        assert [case["case"] for case in result["cases"]] == list(range(1, 11))
        for case in result["cases"]:
            assert case["passed"] and case["rank"] == record["rank"]
            assert case["terminal"] == record["terminal"]
            assert case["input_bits_unchanged"] and case["cumulative_ieee_flags_equal"]
        for name, expected in record["sources"].items():
            assert sha(work/name) == expected
        checked.append({
            "key": record["key"], "vertex": record["vertex"], "rank": record["rank"],
            "terminal": record["terminal"], "flags": record["flags"],
            "result_sha256": sha(result_path), "dump_sha256": result["dump_sha256"],
            "cases": len(result["cases"]),
            "output_coefficients": sum(c["output_coefficients"] for c in result["cases"]),
            "severely_cancelled_coefficients": sum(
                c["severely_cancelled_coefficients"] for c in result["cases"]),
            "fast_cases": sum(c["fast"] for c in result["cases"]),
            "fallback_cases": sum(c["fallback"] for c in result["cases"]),
            "max_new_bound_fraction": max(
                c["max_fraction_of_forward_bound"]["NEW"] for c in result["cases"]),
            "max_new_error_over_epsilon_product_scale": max(
                c["max_error_over_epsilon_product_scale"]["NEW"] for c in result["cases"]),
        })
    output = {
        "schema": "factorization-independent-oracle-campaign-v1",
        "complete": True, "passed": True, "acceptance_eligible": False,
        "scope": "Stored-input algebra, flags and immutability only; no amplitude or timing claim",
        "campaign": str(args.campaign), "campaign_sha256": sha(args.campaign/"run.json"),
        "source_sha256": {name: sha(HERE/name) for name in
                          ("driver.py", "oracle.py", "test_oracle.py", "summarize.py", "README.md")},
        "streams": checked,
        "cases": sum(r["cases"] for r in checked),
        "output_coefficients": sum(r["output_coefficients"] for r in checked),
        "severely_cancelled_coefficients": sum(
            r["severely_cancelled_coefficients"] for r in checked),
        "max_new_bound_fraction": max(r["max_new_bound_fraction"] for r in checked),
        "max_new_error_over_epsilon_product_scale": max(
            r["max_new_error_over_epsilon_product_scale"] for r in checked),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(output, indent=2, sort_keys=True)+"\n")
    print(json.dumps({key: output[key] for key in
                      ("passed", "cases", "output_coefficients",
                       "severely_cancelled_coefficients", "max_new_bound_fraction",
                       "max_new_error_over_epsilon_product_scale")}))


if __name__ == "__main__":
    main()
