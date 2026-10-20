#!/usr/bin/env python3
"""Recheck the source proof using retained files, without physics or builds."""
from fractions import Fraction
import hashlib
import json
from math import comb
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    seal = read(HERE / "proof-seal.json")
    assert seal["count"] == len(seal["files"])
    for name, digest in seal["files"].items():
        assert sha(HERE / name) == digest, name
    result = read(HERE / "analysis.json")
    inputs = read(HERE / "proof-inputs/provenance.json")
    assert sha(HERE / "analyze.py") == result["script_sha256"]
    for record in inputs["files"].values():
        assert sha(HERE / record["retained"]) == record["sha256"]
    ns = runpy.run_path(str(HERE / "analyze.py"), run_name="retained_source_proof")
    inventory = read(HERE / inputs["files"][result["inventory"]["path"]]["retained"])
    assert inputs["files"][result["inventory"]["path"]]["sha256"] == result["inventory"]["sha256"]
    assert inputs["files"][result["trace"]["path"]]["sha256"] == result["trace"]["sha256"]
    proved = excluded = 0
    for case_name, case in result["cases"].items():
        for record in case["vertex_proofs"].values():
            retained = inputs["files"][record["source"]]
            assert retained["sha256"] == record["source_sha256"]
            actual = ns["prove"](HERE / retained["retained"])
            # JSON object keys are strings, including row multiplicities.
            actual = json.loads(json.dumps(actual))
            expected = {k: v for k, v in record.items()
                        if k not in ("source", "source_sha256")}
            assert actual == expected, record["source"]
            proved += actual["passed"]
            excluded += not actual["passed"]
        before = after = 0
        for call in inventory[case_name]["calls"]:
            proof = case["vertex_proofs"].get(call["vertex"], {})
            if not proof.get("passed"):
                continue
            terminal = call["mode"] == "TERMINAL"
            key = ("products_per_coefficient_terminal_all_diagonals" if terminal else
                   "products_per_coefficient_full_per_cut_component")
            factor = comb(call["left"] + 4, 4) * (1 if terminal else call["cut"])
            before += factor * proof[key]["original_after_literal_zeros"]
            after += factor * proof[key]["factored_with_original_b0"]
        total = inventory[case_name]["complex_products_after_literal_zero_skip_before_A_zero_skip"]
        assert before == case["matched_static_products"]
        assert after == case["matched_factored_products"]
        assert total - before + after == case["all_selected_factored_products"]
        assert (before - after) / total == case["static_saved_fraction"]
    checks = read(HERE / "parser-check.json")
    for check in checks["checks"]:
        if check.get("rejected"):
            try:
                ns["expression"](check["unsupported_unary"])
            except AssertionError:
                pass
            else:
                raise AssertionError("Unsupported unary expression accepted")
        else:
            value = ns["expression"](check["input"])
            assert value.terms == {(): (Fraction(check["exact_real"]), 0)}
    assert (proved, excluded) == (16, 2)
    print(json.dumps({"passed": True, "source_maps_proved": proved,
                      "excluded_vector_maps": excluded,
                      "exact_coefficient_identities": 64 * proved,
                      "static_counts_recomputed": True,
                      "retained_files": seal["count"],
                      "numerical_evaluations_started": False}, indent=2))


if __name__ == "__main__":
    main()
