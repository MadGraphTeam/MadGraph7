#!/usr/bin/env python3
"""Exact symbolic source audit of q-dependent Lorentz row factorizations.

No compiler, floating-point amplitude, or Symbolica process is invoked.
"""
import ast
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from math import comb
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
INVENTORY = HERE.parent / "symmetric-accumulation/final-inventory.json"
TRACE = HERE.parent / "symmetric-static-owner/qcd3-operation-trace-v1/operation-summary.json"
PHASES = ((1, 0), (-1, 0), (0, 1), (0, -1))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_pair(a, b):
    return a[0] + b[0], a[1] + b[1]


def mul_pair(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


class Polynomial:
    """Sparse commutative polynomials over exact Gaussian rationals."""
    def __init__(self, terms):
        self.terms = {k: v for k, v in terms.items() if v != (0, 0)}

    def __eq__(self, other):
        return self.terms == other.terms

    def __add__(self, other):
        terms = dict(self.terms)
        for key, value in other.terms.items():
            terms[key] = add_pair(terms.get(key, (0, 0)), value)
        return Polynomial(terms)

    def __mul__(self, other):
        terms = {}
        for a, x in self.terms.items():
            for b, y in other.terms.items():
                key = tuple(sorted(a + b))
                terms[key] = add_pair(terms.get(key, (0, 0)), mul_pair(x, y))
        return Polynomial(terms)

    def __neg__(self):
        return self * constant((-1, 0))

    def __sub__(self, other):
        return self + -other


def constant(value):
    return Polynomial({(): value})


def variable(name):
    return Polynomial({(name,): (1, 0)})


def expression(text):
    # The inventoried coefficients contain only sums/products of scalars.
    text = re.sub(r"([A-Z]\w*)%([A-Z]\w*)\((\d+)\)", r"\1_\2_\3", text)
    text = re.sub(r"([A-Z]\w*)\((\d+)\)", r"\1_\2", text)
    text = re.sub(r"(\d+(?:\.\d*)?)[QD]([+-]?\d+)",
                  lambda m: f"({m[1]}*10**({m[2]}))", text)

    def parse(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return constant((Fraction(str(node.value)), 0))
        if isinstance(node, ast.Name):
            return constant((0, 1)) if node.id == "CI" else variable(node.id)
        if isinstance(node, ast.UnaryOp):
            operand = parse(node.operand)
            return -operand if isinstance(node.op, ast.USub) else operand
        if isinstance(node, ast.BinOp):
            a, b = parse(node.left), parse(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Pow):
                # Only the decimal constant normalization above uses powers.
                assert not any(a.terms.keys()) and not any(b.terms.keys())
                base, imag = a.terms.get((), (0, 0))
                power, power_imag = b.terms.get((), (0, 0))
                assert imag == power_imag == 0 and power.denominator == 1
                return constant((base ** int(power), 0))
        raise AssertionError(ast.dump(node))

    return parse(ast.parse(text, mode="eval").body)


def logical(path):
    rows = []
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw[0] in "cC*!":
            continue
        body = re.sub(r"\s+", "", raw[6:]).upper()
        if raw[5] not in " 0":
            rows[-1] += body
        else:
            rows.append(body)
    return rows


def assignments(path):
    result, original = {}, {}
    for line in logical(path):
        match = re.fullmatch(r"COEFF\((\d+),(\d+),(\d+)\)=(.*)", line)
        if match:
            key = tuple(map(int, match.groups()[:3]))
            assert key not in result
            original[key] = match[4]
            result[key] = expression(match[4])
    return result, original


def prove(path):
    coeff, original = assignments(path)
    zero = constant((0, 0))
    q_rows = []
    for m in range(1, 5):
        mapping = []
        for j in range(1, 5):
            target = [coeff[j, m, k] for k in range(1, 5)]
            matches = [(r, phase) for r in range(1, 5) for phase in PHASES
                       if target == [constant(phase) * coeff[r, 1, k] for k in range(1, 5)]]
            if not matches:
                return {"passed": False, "unmatched": {"row": j, "q_index": m}}
            assert len(matches) == 1, "Degenerate source row requires explicit treatment"
            r, phase = matches[0]
            mapping.append({"source_row": r, "phase": list(phase)})
        assert sorted(item["source_row"] for item in mapping) == [1, 2, 3, 4]
        q_rows.append(mapping)
    b0_nonzero = [sum(coeff[j, 0, k] != zero for k in range(1, 5)) for j in range(1, 5)]
    h_nonzero = [sum(coeff[j, 1, k] != zero for k in range(1, 5)) for j in range(1, 5)]
    terminal_rows = [sorted({q_rows[m][j]["source_row"] for m in range(4)})
                     for j in range(4)]
    multiplicities = []
    for j in range(4):
        counts = defaultdict(int)
        for m in range(4):
            counts[q_rows[m][j]["source_row"]] += 1
        multiplicities.append(dict(counts))
    original_full = sum(b0_nonzero) + 4 * sum(h_nonzero)
    factored_full = sum(b0_nonzero) + sum(h_nonzero)
    factored_terminal = sum(b0_nonzero) + sum(
        h_nonzero[row-1] for selected in terminal_rows for row in selected)
    return {
        "passed": True, "coefficient_identities_checked": 64,
        "q_rows": q_rows, "terminal_required_u_rows": terminal_rows,
        "terminal_row_multiplicities": multiplicities,
        "b0_nonzero_by_row": b0_nonzero, "h_nonzero_by_row": h_nonzero,
        "products_per_coefficient_full_per_cut_component": {
            "original_after_literal_zeros": original_full,
            "factored_with_original_b0": factored_full},
        "products_per_coefficient_terminal_all_diagonals": {
            "original_after_literal_zeros": original_full,
            "factored_with_original_b0": factored_terminal},
        "h_source_expressions": {
            f"{j},{k}": original[j, 1, k]
            for j in range(1, 5) for k in range(1, 5) if coeff[j, 1, k] != zero},
    }


def main():
    inventory = json.loads(INVENTORY.read_text())
    output = {"schema": "symbolic-q-row-factorization-v1", "numerical_evaluation": False,
              "source_revision": "b44d20e725f1ec68589df20284e4ce9116e9e2ab",
              "inventory": {"path": str(INVENTORY.relative_to(ROOT)), "sha256": sha(INVENTORY)},
              "trace": {"path": str(TRACE.relative_to(ROOT)), "sha256": sha(TRACE)},
              "script_sha256": sha(Path(__file__)), "cases": {}}
    for case in ("qcd-3", "qcd-4"):
        source = inventory[case]
        assert source["generation_revision"] == output["source_revision"]
        proofs = {}
        for vertex, routine in source["routines"].items():
            if not any(s["in_size"] == s["out_size"] == 4 for s in routine["shapes"]):
                continue
            path = ROOT / source["source_output"] / "Source/DHELAS" / f"{vertex}.f"
            assert sha(path) == source["sources"][str(path.relative_to(ROOT))]
            proofs[vertex] = {"source": str(path.relative_to(ROOT)), "source_sha256": sha(path),
                              **prove(path)}
        baseline, factored = 0, 0
        by_kind = defaultdict(lambda: {"original_products": 0, "factored_products": 0, "calls": 0})
        for call in source["calls"]:
            proof = proofs.get(call["vertex"], {})
            if not proof.get("passed"):
                continue
            terminal = call["mode"] == "TERMINAL"
            numbers = proof["products_per_coefficient_terminal_all_diagonals" if terminal else
                            "products_per_coefficient_full_per_cut_component"]
            factor = comb(call["left"] + 4, 4) * (1 if terminal else call["cut"])
            before, after = (factor * numbers[k] for k in
                             ("original_after_literal_zeros", "factored_with_original_b0"))
            baseline += before
            factored += after
            key = f"{call['mode']}:{call['left']}"
            by_kind[key]["original_products"] += before
            by_kind[key]["factored_products"] += after
            by_kind[key]["calls"] += 1
        total = source["complex_products_after_literal_zero_skip_before_A_zero_skip"]
        output["cases"][case] = {
            "vertex_proofs": proofs, "matched_static_products": baseline,
            "matched_factored_products": factored,
            "all_selected_static_products": total,
            "all_selected_factored_products": total - baseline + factored,
            "static_saved_fraction": (baseline - factored) / total,
            "by_kind_rank": dict(by_kind),
            "scope": "Literal vertex zeros removed; before runtime A/B zero skips. Not timings."}
    trace = json.loads(TRACE.read_text())
    actual_bounds = []
    for helper in trace["helpers"]:
        proof = output["cases"]["qcd-3"]["vertex_proofs"].get(helper["vertex"], {})
        if not proof.get("passed"):
            continue
        terminal = helper["terminal"]
        nc = comb(helper["rank"] + 4, 4)
        cut = 1 if terminal else 4
        constant_ceiling = sum(proof["b0_nonzero_by_row"]) * nc * cut * helper["fast_calls"]
        count = helper["counts"]["original_products_after_A_B_zero_skips"]
        repetition = 2 if terminal else 4
        if terminal:
            assert all(set(v.values()) == {2}
                       for v in proof["terminal_row_multiplicities"])
        lower = max(Fraction(0), Fraction(count - constant_ceiling) *
                    Fraction(repetition - 1, repetition))
        upper = Fraction(count) * Fraction(repetition - 1, repetition)
        actual_bounds.append({
            "helper_index": helper["index"], "vertex": helper["vertex"],
            "rank": helper["rank"], "terminal": terminal,
            "original_products_after_actual_zero_skips": count,
            "unchanged_b0_product_ceiling": constant_ceiling,
            "saved_products_lower_bound": str(lower),
            "saved_products_upper_bound": str(upper),
            "scope": "Conditional on the exact runtime q-row relation guard passing. "
                     "Existing trace lacks B0/q split; bounds, not measured factorization counts."})
    output["actual_qcd3_trace_bounds"] = actual_bounds
    target = HERE / "analysis.json"
    assert not target.exists(), "Retain earlier source-analysis evidence"
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    for case, data in output["cases"].items():
        print(case, "proved", [k for k, v in data["vertex_proofs"].items() if v["passed"]],
              "static saved", data["static_saved_fraction"])


if __name__ == "__main__":
    main()
