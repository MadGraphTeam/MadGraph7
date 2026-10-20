#!/usr/bin/env python3
"""Exact source identity and symbolic-support counts; no kernel or amplitude."""
import sys
sys.dont_write_bytecode = True

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = next(p for p in HERE.parents if (p/"madgraph/__init__.py").is_file())
PARSER_SHA256 = "ce7c060c6c3f6a556ad1b518800030091dcc226a422bd17c1ef3ec5afe6d039a"
PROOF_SHA256 = "f5bc8c65680a028e79103f788fefc2356962350a9a7cffd8ffacbfad49510f42"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_parser():
    path = PARENT/"analyze.py"
    assert sha(path) == PARSER_SHA256
    spec = importlib.util.spec_from_file_location("sealed_factor_source_parser", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matrix_strings(vertex):
    if vertex == "MP_FFV1L1_2":
        return [
            ["P2(0)-P2(3)", "-P2(1)+CI*P2(2)", "-M2", "0"],
            ["-P2(1)-CI*P2(2)", "P2(0)+P2(3)", "0", "-M2"],
            ["-M2", "0", "P2(0)+P2(3)", "P2(1)-CI*P2(2)"],
            ["0", "-M2", "P2(1)+CI*P2(2)", "P2(0)-P2(3)"],
        ]
    if vertex == "MP_FFV1L2_1":
        return [
            ["P1(0)+P1(3)", "P1(1)+CI*P1(2)", "M1", "0"],
            ["P1(1)-CI*P1(2)", "P1(0)-P1(3)", "0", "M1"],
            ["M1", "0", "P1(0)-P1(3)", "-P1(1)-CI*P1(2)"],
            ["0", "M1", "-P1(1)+CI*P1(2)", "P1(0)+P1(3)"],
        ]
    raise ValueError(vertex)


def substitute(parser, polynomial, replacements):
    result = parser.constant((0, 0))
    for variables, coefficient in polynomial.terms.items():
        term = parser.constant(coefficient)
        for variable in variables:
            term = term * replacements.get(variable, parser.variable(variable))
        result = result + term
    return result


def support_counts(parser, coefficients, r_matrix, needed, mask):
    zero = parser.constant((0, 0))
    active_k = [k for k in range(1, 5) if mask & (1 << (k-1))]
    h_products = {
        r: sum(coefficients[r, 1, k] != zero for k in active_k)
        for r in range(1, 5)
    }
    direct_by_j = {
        j: sum(coefficients[j, 0, k] != zero for k in active_k)
        for j in range(1, 5)
    }
    r_rows = {
        j: {r for r in range(1, 5) if r_matrix[j-1][r-1] != zero}
        for j in range(1, 5)
    }
    r_products = {
        j: sum(h_products[r] != 0 for r in r_rows[j]) for j in range(1, 5)
    }
    full_u = sum(h_products.values())
    full = {
        "existing_q_u_products": full_u,
        "existing_direct_b0_products": sum(direct_by_j.values()),
        "proposed_r_u_products": sum(r_products.values()),
    }
    full["existing_total"] = full_u + full["existing_direct_b0_products"]
    full["proposed_total"] = full_u + full["proposed_r_u_products"]
    terminal = []
    for j in range(1, 5):
        old_u = sum(h_products[r] for r in needed[j-1])
        new_rows = set(needed[j-1]) | r_rows[j]
        new_u = sum(h_products[r] for r in new_rows)
        terminal.append({
            "diagonal": j,
            "existing_q_u_rows": needed[j-1],
            "proposed_q_and_b0_u_rows": sorted(new_rows),
            "existing_q_u_products": old_u,
            "additional_u_products": new_u-old_u,
            "existing_direct_b0_products": direct_by_j[j],
            "proposed_r_u_products": r_products[j],
            "existing_total": old_u+direct_by_j[j],
            "proposed_total": new_u+r_products[j],
        })
    return {
        "active_a_components": active_k, "full_per_cut_component": full,
        "terminal_per_diagonal": terminal,
        "terminal_all_diagonals_if_same_mask": {
            "existing_total": sum(v["existing_total"] for v in terminal),
            "proposed_total": sum(v["proposed_total"] for v in terminal),
        },
    }


def prove(parser, vertex, proof):
    path = PARENT/"proof-inputs/qcd-3"/(vertex+".f")
    assert sha(path) == proof["source_sha256"]
    coefficients, _ = parser.assignments(path)
    assert len(coefficients) == 80
    strings = matrix_strings(vertex)
    matrix = [[parser.expression(value) for value in row] for row in strings]
    zero = parser.constant((0, 0))
    checked = []
    for j in range(1, 5):
        for k in range(1, 5):
            expected = zero
            for r in range(1, 5):
                expected = expected + matrix[j-1][r-1] * coefficients[r, 1, k]
            assert expected == coefficients[j, 0, k], (vertex, j, k)
            checked.append([j, k])
    outgoing = "P2" if vertex == "MP_FFV1L1_2" else "P1"
    incoming = "P1" if vertex == "MP_FFV1L1_2" else "P2"
    mass = "M2" if vertex == "MP_FFV1L1_2" else "M1"
    width = "W2" if vertex == "MP_FFV1L1_2" else "W1"
    source_lines = parser.logical(path)
    routing = [f"{outgoing}({mu})=+{incoming}({mu})+V3%P({mu})" for mu in range(4)]
    assert all(line in source_lines for line in routing)
    all_variables = set(v for value in coefficients.values() for term in value.terms for v in term)
    assert width not in all_variables
    scenarios = {
        "generic_massive": {},
        "massless": {mass: zero},
        "massive_longitudinal": {outgoing+"_1": zero, outgoing+"_2": zero},
        "massless_longitudinal": {
            mass: zero, outgoing+"_1": zero, outgoing+"_2": zero},
        "massive_rest": {
            outgoing+"_1": zero, outgoing+"_2": zero, outgoing+"_3": zero},
        "massless_plus_z": {
            mass: zero, outgoing+"_1": zero, outgoing+"_2": zero,
            outgoing+"_3": parser.variable(outgoing+"_0")},
    }
    counts = {}
    for name, replacements in scenarios.items():
        c = {key: substitute(parser, value, replacements)
             for key, value in coefficients.items()}
        r = [[substitute(parser, value, replacements) for value in row] for row in matrix]
        counts[name] = [
            support_counts(parser, c, r, proof["terminal_required_u_rows"], mask)
            for mask in range(16)]
    return {
        "source": str(path.relative_to(ROOT)), "source_sha256": sha(path),
        "b0_equals_r_times_h": True, "identities_checked": checked,
        "h_definition": "H(j,k)=COEFF(j,1,k)", "r_matrix": strings,
        "routing_checked": routing, "required_arguments": [outgoing+"(0:3)", mass],
        "mass_declaration": "REAL*16", "momentum_declaration": "COMPLEX*32",
        "width_not_in_coefficient_expressions": width,
        "symbolic_support_counts": counts,
    }


def main():
    parser = load_parser()
    proof_path = PARENT/"analysis.json"
    assert sha(proof_path) == PROOF_SHA256
    proofs = json.loads(proof_path.read_text())["cases"]["qcd-3"]["vertex_proofs"]
    vertices = ("MP_FFV1L1_2", "MP_FFV1L2_1")
    output = {
        "schema": "b0-source-identity-and-support-counts-v1",
        "source_revision": "b44d20e725f1ec68589df20284e4ce9116e9e2ab",
        "numerical_evaluation": False, "prototype_implemented": False,
        "script_sha256": sha(Path(__file__)), "parser_sha256": PARSER_SHA256,
        "derivative_proof_sha256": PROOF_SHA256,
        "count_scope": (
            "Complex/scalar product slots, ignoring different unit costs. Source literal zeros "
            "and explicit symbolic mass/momentum substitutions are removed. Each of16 A-support "
            "masks is counted; U is assumed zero-pruned when structurally zero. Accidental numerical "
            "zeros/cancellation, runtime guards, additions and coefficient-formation costs are excluded. "
            "These are conditional structural counts, not physical trace counts or speedups."),
        "vertices": {vertex: prove(parser, vertex, proofs[vertex]) for vertex in vertices},
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
