#!/usr/bin/env python3
"""Read-only rehash and optional exact replay of retained oracle campaigns."""
import sys
sys.dont_write_bytecode = True

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import oracle

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/"madgraph/__init__.py").is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value):
    return json.loads(json.dumps(value, sort_keys=True))


def verify_campaign(path, recompute=False):
    aggregate = json.loads(path.read_text())
    assert aggregate["schema"] == "factorization-independent-oracle-campaign-v1"
    assert aggregate["complete"] and aggregate["passed"]
    for name, digest in aggregate["source_sha256"].items():
        assert sha(HERE/name) == digest, (path, name)
    campaign = ROOT/aggregate["campaign"]
    run_path = campaign/"run.json"
    assert sha(run_path) == aggregate["campaign_sha256"]
    run = json.loads(run_path.read_text())
    assert run["complete"] and len(run["results"]) == len(aggregate["streams"])
    assert run["factor_sha256"] == sha(campaign/"factor.py")
    assert run["runner_sha256"] == sha(campaign/"run_oracle.py")
    assert run["oracle_driver_sha256"] == sha(campaign/"oracle-driver.py")
    assert run["oracle_driver_sha256"] == sha(HERE/"driver.py")
    proof_path = HERE.parent/"analysis.json"
    assert sha(proof_path) == run["proof_sha256"] == oracle.PROOF_SHA256
    proofs = json.loads(proof_path.read_text())["cases"]["qcd-4"]["vertex_proofs"]
    unique = set()
    cases = outputs = cancelled = generated = 0
    bound_fraction = normalized_error = 0.0
    for entry, summary in zip(run["results"], aggregate["streams"], strict=True):
        assert entry["key"] == summary["key"] and entry["key"] not in unique
        unique.add(entry["key"])
        work = campaign/entry["key"]
        result_path = work/"exact-oracle.json"
        assert sha(result_path) == summary["result_sha256"]
        result = json.loads(result_path.read_text())
        assert result["complete"] and result["passed"]
        assert entry["returncode"] == entry["build"]["returncode"] == 0
        assert result["vertex"] == entry["vertex"] == summary["vertex"]
        assert result["oracle_sha256"] == sha(HERE/"oracle.py")
        assert result["proof_sha256"] == oracle.PROOF_SHA256
        dump = work/"dump.txt.gz"
        assert (sha(dump) == entry["gzip_sha256"] == result["dump_sha256"]
                == summary["dump_sha256"])
        assert hashlib.sha256(gzip.decompress(dump.read_bytes())).hexdigest() == entry["raw_dump_sha256"]
        for name, expected in entry["sources"].items():
            assert sha(work/name) == expected, (work, name)
            generated += 1
        # Executables may be omitted from the retained textual evidence package.
        # When present, verify their independently recorded identity too.
        if (work/"oracle-check").exists():
            assert sha(work/"oracle-check") == entry["executable_sha256"]
        assert entry["rank"] == summary["rank"]
        assert entry["terminal"] == summary["terminal"]
        assert entry["flags"] == summary["flags"]
        assert [c["case"] for c in result["cases"]] == list(range(1, 11))
        if recompute:
            exact = [oracle.check_case(c, proofs[entry["vertex"]])
                     for c in oracle.read_cases(dump)]
            assert normalize(exact) == result["cases"], (work, "exact replay")
        current = result["cases"]
        for case in current:
            assert case["passed"] and case["input_bits_unchanged"]
            assert case["cumulative_ieee_flags_equal"]
            assert case["rank"] == entry["rank"] and case["terminal"] == entry["terminal"]
        assert summary["cases"] == len(current)
        assert summary["output_coefficients"] == sum(c["output_coefficients"] for c in current)
        assert summary["severely_cancelled_coefficients"] == sum(
            c["severely_cancelled_coefficients"] for c in current)
        assert summary["fast_cases"] == sum(c["fast"] for c in current)
        assert summary["fallback_cases"] == sum(c["fallback"] for c in current)
        assert summary["max_new_bound_fraction"] == max(
            c["max_fraction_of_forward_bound"]["NEW"] for c in current)
        assert summary["max_new_error_over_epsilon_product_scale"] == max(
            c["max_error_over_epsilon_product_scale"]["NEW"] for c in current)
        cases += summary["cases"]
        outputs += summary["output_coefficients"]
        cancelled += summary["severely_cancelled_coefficients"]
        bound_fraction = max(bound_fraction, summary["max_new_bound_fraction"])
        normalized_error = max(normalized_error, summary["max_new_error_over_epsilon_product_scale"])
    assert aggregate["cases"] == cases and aggregate["output_coefficients"] == outputs
    assert aggregate["severely_cancelled_coefficients"] == cancelled
    assert aggregate["max_new_bound_fraction"] == bound_fraction
    assert aggregate["max_new_error_over_epsilon_product_scale"] == normalized_error
    return dict(aggregate=str(path.relative_to(ROOT)), streams=len(unique), cases=cases,
                complex_outputs=outputs, generated_sources=generated,
                exact_recomputed=recompute, passed=True)


def verify_guards(campaign):
    data = json.loads((campaign/"results.json").read_text())
    assert data["complete"] and len(data["results"]) == 8
    assert data["comparisons"] == 152
    assert data["factor_sha256"] == sha(campaign/"factor.py")
    assert data["runner_sha256"] == sha(campaign/"check_true_guards.py")
    assert data["oracle_driver_sha256"] == sha(HERE/"driver.py")
    seen = set()
    generated = 0
    for record in data["results"]:
        assert record["key"] not in seen
        seen.add(record["key"])
        assert record["build"]["returncode"] == record["run"]["returncode"] == 0
        lines = [line.split() for line in record["run"]["output"].splitlines()]
        assert lines == [["PASS", str(i)] for i in range(1, 20)]
        work = campaign/record["key"]
        assert (work/"run.log").read_text() == record["run"]["output"]
        for name, expected in record["sources"].items():
            assert sha(work/name) == expected, (work, name)
            generated += 1
        if (work/"guards").exists():
            assert sha(work/"guards") == record["executable_sha256"]
    return dict(campaign=str(campaign.relative_to(ROOT)), comparisons=152,
                generated_sources=generated, passed=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregates", type=Path, nargs="*")
    parser.add_argument("--recompute", action="store_true",
                        help="Repeat exact Fraction checks from the raw retained dumps")
    parser.add_argument("--guards", type=Path, action="append", default=[])
    args = parser.parse_args()
    paths = args.aggregates or sorted(HERE.glob("validation-*.json"))
    assert paths, "No campaigns selected"
    result = dict(passed=True, read_only=True, symbolica_processes=0,
                  campaigns=[verify_campaign(p.resolve(), args.recompute) for p in paths],
                  guard_campaigns=[verify_guards(p.resolve()) for p in args.guards])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
