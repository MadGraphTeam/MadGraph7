#!/usr/bin/env python3
"""Exact stored-binary64 contraction oracle; no amplitude/library evaluation.

Adapted in a separate experimental namespace from the frozen binary128 oracle.
The decoder, range contract, fixtures and proof binding are binary64-specific.
"""
import sys
sys.dont_write_bytecode = True

import argparse
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import itertools
import json
from math import comb
from pathlib import Path
import re

EPS = Fraction(1, 2**52)
ZERO = (Fraction(0), Fraction(0))
INHERITED_PROOF_SHA256 = "f5bc8c65680a028e79103f788fefc2356962350a9a7cffd8ffacbfad49510f42"


def binary64(word):
    """Decode IEEE binary64 using integer/rational arithmetic, never float."""
    assert re.fullmatch(r"[0-9a-fA-F]{16}", word)
    bits = int(word, 16)
    sign = -1 if bits >> 63 else 1
    exponent = (bits >> 52) & 0x7ff
    significand = bits & ((1 << 52) - 1)
    if exponent == 0x7ff:
        raise ValueError("Nonfinite input/output belongs to the safety suite")
    if exponent:
        significand += 1 << 52
        power = exponent - 1023 - 52
    else:
        power = 1 - 1023 - 52
    value = Fraction(sign * significand)
    value = value * (1 << power) if power >= 0 else value / (1 << -power)
    return value, bits


def complex64(words):
    assert len(words) == 2
    real, real_bits = binary64(words[0])
    imag, imag_bits = binary64(words[1])
    return (real, imag), (real_bits, imag_bits)


def admitted_bits(bits):
    """Signed zero, or a finite normal component with unbiased e in[-200,200]."""
    magnitude = bits & ((1 << 63)-1)
    exponent = (bits >> 52) & 0x7ff
    return magnitude == 0 or 823 <= exponent <= 1223


def multiply(a, b):
    return a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0]


def add(a, b):
    return a[0]+b[0], a[1]+b[1]


def scale_products(a, b):
    """L1 sum of the four exact real products in one complex multiplication."""
    return abs(a[0]*b[0]) + abs(a[1]*b[1]) + abs(a[0]*b[1]) + abs(a[1]*b[0])


def position(monomial):
    if not monomial:
        return 0
    return comb(len(monomial)+3, 4) + sum(
        comb(index+i, i+1) for i, index in enumerate(sorted(monomial)) if index)


def monomials(rank):
    """Independent enumeration of the symmetric monomial index contract."""
    result = {}
    for degree in range(rank+1):
        for monomial in itertools.combinations_with_replacement(range(4), degree):
            index = position(monomial)
            assert index not in result
            result[index] = monomial
    assert set(result) == set(range(comb(rank+4, 4)))
    return result


def read_cases(path):
    opener = gzip.open if path.suffix == ".gz" else open
    current = None
    cases = []
    layout = None
    passed = False
    format_seen = False
    with opener(path, "rt") as source:
        for lineno, line in enumerate(source, 1):
            values = line.split()
            if not values:
                continue
            assert not passed, "Records after completion marker"
            tag = values[0]
            if tag == "FORMAT":
                assert not format_seen and layout is None and current is None
                assert values == ["FORMAT", "BINARY64_HEX_V1"]
                format_seen = True
            elif tag == "LAYOUT":
                assert format_seen and len(values) == 4
                assert layout is None and list(map(int, values[1:])) == [64, 53, 1024]
                layout = list(map(int, values[1:]))
            elif tag == "CASE":
                assert layout is not None and current is None
                number, rank, terminal, cut, ins, outs, diagonal_only = map(int, values[1:])
                assert number == len(cases)+1 and 2 <= rank <= 4 and terminal in (0, 1)
                assert cut == ins == outs == 4
                assert diagonal_only == int(bool(terminal) and number != 8)
                current = dict(number=number, rank=rank, terminal=bool(terminal),
                               diagonal_only=bool(diagonal_only),
                               cut=cut, ins=ins, outs=outs, arrays={}, bits={},
                               flags={}, immutable={}, tail={}, trace=None)
            elif tag in ("A", "B", "OLD", "NEW"):
                assert current is not None
                assert len(values) == 6
                key = tuple(map(int, values[1:4]))
                array = current["arrays"].setdefault(tag, {})
                bits = current["bits"].setdefault(tag, {})
                assert key not in array, (path, lineno, tag, key)
                array[key], bits[key] = complex64(values[4:])
            elif tag == "FLAGS":
                assert current is not None and values[1] in ("OLD", "NEW")
                assert values[1] not in current["flags"]
                flags = tuple(map(int, values[2:]))
                assert len(flags) >= 5 and set(flags).issubset({0, 1})
                current["flags"][values[1]] = flags
            elif tag == "IMMUTABLE":
                assert current is not None and values[1] in ("OLD", "NEW")
                assert values[1] not in current["immutable"]
                current["immutable"][values[1]] = tuple(map(int, values[2:]))
            elif tag == "TAIL_ZERO":
                assert current is not None and values[1] in ("OLD", "NEW")
                assert values[1] not in current["tail"]
                current["tail"][values[1]] = int(values[2])
            elif tag == "TRACE":
                assert current is not None and current["trace"] is None
                current["trace"] = tuple(map(int, values[1:]))
            elif tag == "ENDCASE":
                assert current is not None and values == ["ENDCASE"]
                cases.append(current)
                current = None
            elif tag == "PASS_DUMP":
                assert values == ["PASS_DUMP"] and current is None and not passed
                passed = True
            else:
                raise ValueError(f"Unknown dump record {path}:{lineno}: {line.rstrip()}")
    assert current is None and len(cases) == 10 and layout is not None and passed
    return cases


def expected_arrays(case):
    rank = case["rank"]
    nc = comb(rank+4, 4)
    no = comb(rank+5, 4)
    return {
        "A": {(k, l, i) for k in range(1, 5) for l in range(nc) for i in range(1, 5)},
        "B": {(j, m, k) for j in range(1, 5) for m in range(5) for k in range(1, 5)},
        "OLD": {(j, n, i) for j in range(1, 5) for n in range(no) for i in range(1, 5)},
        "NEW": {(j, n, i) for j in range(1, 5) for n in range(no) for i in range(1, 5)},
    }


def check_rows(case, proof):
    coefficients = case["arrays"]["B"]
    for m, rows in enumerate(proof["q_rows"], 1):
        for j, row in enumerate(rows, 1):
            phase = tuple(map(Fraction, row["phase"]))
            for k in range(1, 5):
                if coefficients[j, m, k] != multiply(phase, coefficients[row["source_row"], 1, k]):
                    return False
    return True


def check_case(case, proof):
    for name, expected in expected_arrays(case).items():
        assert set(case["arrays"].get(name, {})) == expected, (case["number"], name)
    assert case["immutable"] == {"OLD": (1, 1), "NEW": (1, 1)}, case["number"]
    assert case["tail"] == {"OLD": 1, "NEW": 1}
    assert set(case["flags"]) == {"OLD", "NEW"}
    assert case["flags"]["OLD"] == case["flags"]["NEW"], case["number"]
    fast, fallback = case["trace"]
    assert fast + fallback == 1 and min(fast, fallback) >= 0
    if case["number"] == 8:
        assert (fast, fallback) == (0, 1), "False entry-inexact must use original"
    elif case["number"] != 7:
        assert (fast, fallback) == (1, 0), "Guard-compatible controlled fixture must be exercised"
    row_relation = check_rows(case, proof)
    assert not fast or row_relation
    admitted_inputs = all(admitted_bits(bits) for tag in ("A", "B")
                          for value in case["bits"][tag].values() for bits in value)
    assert not fast or admitted_inputs
    zero_components = Counter()
    for key, value in case["arrays"]["B"].items():
        if key[1] == 0 or value == ZERO:
            continue
        for component in range(2):
            if value[component] == 0:
                label = ("real" if component == 0 else "imag")
                label += "_negative_zero" if case["bits"]["B"][key][component] >> 63 else "_positive_zero"
                zero_components[label] += 1
    if case["number"] in (9, 10):
        assert all(zero_components[label] > 0 for label in (
            "real_negative_zero", "real_positive_zero", "imag_negative_zero", "imag_positive_zero"))
    surviving_perturbations = None
    if case["number"] == 1:
        surviving_perturbations = sum(
            value != (Fraction((17*k+11*l+7*i) % 31-15, 16),
                      Fraction((13*k+5*l+19*i) % 29-14, 32))
            for (k, l, i), value in case["arrays"]["A"].items())
        assert surviving_perturbations > 0, "Binary64-scale perturbations must survive storage"
    a, b = case["arrays"]["A"], case["arrays"]["B"]
    input_monomials = monomials(case["rank"])
    totals = {key: ZERO for key in case["arrays"]["OLD"]}
    scales = {key: Fraction(0) for key in totals}
    counts = Counter()
    # Reconstruct the full mathematical sum directly from stored A and B.
    # This does not reuse the prototype's U expressions or row map.
    for j in range(1, 5):
        for m in range(5):
            for k in range(1, 5):
                for i in range(1, 5):
                    for l, monomial in input_monomials.items():
                        n = position(monomial + (() if m == 0 else (m-1,)))
                        key = j, n, i
                        av, bv = a[k, l, i], b[j, m, k]
                        totals[key] = add(totals[key], multiply(av, bv))
                        scales[key] += scale_products(av, bv)
                        if av != ZERO and bv != ZERO:
                            counts[key] += 1
    worst = {"OLD": Fraction(0), "NEW": Fraction(0)}
    worst_gamma_fraction = {"OLD": Fraction(0), "NEW": Fraction(0)}
    worst_records = {"OLD": None, "NEW": None}
    differences, zero_sign_differences, projected_components = 0, 0, 0
    cancelled = 0
    for key, reference in totals.items():
        scale = scales[key]
        if scale and abs(reference[0])+abs(reference[1]) < scale / 2**35:
            cancelled += 1
        # Conservative forward bound for <=20 complex terms, including product
        # rounding and either the original or grouped-U accumulation. Error is
        # normalized by the absolute real-product scale, not by a cancelled sum.
        for name in ("OLD", "NEW"):
            # Original DP has only the full public updater. A future accepted
            # terminal prototype promises the diagonal and explicit zeros
            # elsewhere; rejected calls must return the full original result.
            projected = name == "NEW" and case["terminal"] and fast and key[0] != key[2]
            expected = ZERO if projected else reference
            active_scale = Fraction(0) if projected else scale
            active_count = 0 if projected else counts[key]
            steps = 2*active_count+8
            gamma = steps*EPS/(1-steps*EPS)
            bound = gamma*active_scale
            value = case["arrays"][name][key]
            error = max(abs(value[0]-expected[0]), abs(value[1]-expected[1]))
            assert error <= bound, (case["number"], name, key, str(error), str(bound))
            if active_scale:
                normalized = error/(EPS*active_scale)
                if worst_records[name] is None or normalized > worst[name]:
                    worst_records[name] = {
                        "index": key, "reference_real": str(expected[0]),
                        "reference_imag": str(expected[1]),
                        "absolute_error": str(error), "product_scale": str(active_scale),
                        "nonzero_complex_terms": active_count, "bound": str(bound),
                        "observed_real": str(value[0]), "observed_imag": str(value[1]),
                    }
                worst[name] = max(worst[name], normalized)
                worst_gamma_fraction[name] = max(worst_gamma_fraction[name], error/bound)
        old, new = case["arrays"]["OLD"][key], case["arrays"]["NEW"][key]
        if case["terminal"] and fast and key[0] != key[2]:
            projected_components += 1
            continue
        differences += old != new
        for component in range(2):
            if old[component] == new[component] == 0:
                zero_sign_differences += (case["bits"]["OLD"][key][component] >>
                                         63 != case["bits"]["NEW"][key][component] >> 63)
    if not fast:
        assert case["bits"]["OLD"] == case["bits"]["NEW"], "Fallback must be bit-identical"
    return {
        "case": case["number"], "rank": case["rank"], "terminal": case["terminal"],
        "fast": fast, "fallback": fallback, "exact_stored_row_relation": row_relation,
        "all_active_inputs_within_exponent_contract": admitted_inputs,
        "nonzero_b_zero_component_counts": dict(zero_components),
        "surviving_binary64_perturbations": surviving_perturbations,
        "output_coefficients": len(totals), "severely_cancelled_coefficients": cancelled,
        "original_reference": "unchanged full public DP updater",
        "terminal_projection_zero_coefficients": projected_components,
        "different_numeric_outputs": differences,
        "different_zero_sign_components": zero_sign_differences,
        "max_error_over_epsilon_product_scale": {k: float(v) for k, v in worst.items()},
        "max_fraction_of_forward_bound": {k: float(v) for k, v in worst_gamma_fraction.items()},
        "worst_normalized_error_exact": worst_records,
        "input_bits_unchanged": True, "cumulative_ieee_flags_equal": True,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument("--vertex", required=True)
    parser.add_argument("--proof", type=Path, default=Path(__file__).resolve().parent/"source-parity.json")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.result.exists():
        raise RuntimeError("Refuse to overwrite an oracle result")
    data = json.loads(args.proof.read_text())
    assert data["complete"] and data["passed"]
    assert data["inherited_proof_sha256"] == INHERITED_PROOF_SHA256
    proof = data["vertices"][args.vertex]
    assert proof["passed"]
    results = []
    output = {"schema": "exact-binary64-contraction-oracle-v1", "complete": False,
              "passed": False, "vertex": args.vertex,
              "dump_sha256": hashlib.sha256(args.dump.read_bytes()).hexdigest(),
              "proof_sha256": hashlib.sha256(args.proof.read_bytes()).hexdigest(),
              "oracle_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "arithmetic": "Exact Python Fraction over decoded stored IEEE binary64 A/B",
              "epsilon": "2^-52", "cancellation_reporting_threshold": "2^-35",
              "input_exponent_contract": "zero or unbiased exponent[-200,200] per component",
              "forward_bound": "gamma_(2*N+8)(epsilon) * sum of absolute real products",
              "cases": results}
    try:
        for case in read_cases(args.dump):
            results.append(check_case(case, proof))
        output.update(complete=True, passed=True)
    except Exception as error:
        output["failure"] = repr(error)
        raise
    finally:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(output, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"passed": True, "cases": len(results),
                      "output_coefficients": sum(r["output_coefficients"] for r in results),
                      "maximum_new_bound_fraction": max(r["max_fraction_of_forward_bound"]["NEW"]
                                                       for r in results)}))


if __name__ == "__main__":
    main()
