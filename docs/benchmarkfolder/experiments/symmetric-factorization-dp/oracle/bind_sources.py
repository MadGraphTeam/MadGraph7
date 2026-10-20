#!/usr/bin/env python3
"""Bind retained actual DP expressions to exact proved derivative-row maps.

This source audit performs only exact symbolic algebra. It never builds or runs
a numerical helper. Run after the coordinated quiet window is released.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
QP = HERE.parent.parent/"symmetric-factorization"
ROOT = next(p for p in HERE.parents if (p/"madgraph/__init__.py").is_file())
REVISION = "b44d20e725f1ec68589df20284e4ce9116e9e2ab"
PARSER_SHA256 = "ce7c060c6c3f6a556ad1b518800030091dcc226a422bd17c1ef3ec5afe6d039a"
PROOF_SHA256 = "f5bc8c65680a028e79103f788fefc2356962350a9a7cffd8ffacbfad49510f42"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind():
    parser_path = QP/"analyze.py"
    assert sha(parser_path) == PARSER_SHA256
    spec = importlib.util.spec_from_file_location("sealed_exact_source_parser", parser_path)
    source_parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source_parser)
    proof_path = QP/"analysis.json"
    assert sha(proof_path) == PROOF_SHA256
    proof_data = json.loads(proof_path.read_text())
    assert proof_data["source_revision"] == REVISION
    result = {
        "schema": "binary64-actual-source-map-parity-v1", "complete": True,
        "passed": True, "source_revision": REVISION, "numerical_evaluation": False,
        "script_sha256": sha(Path(__file__)), "parser_sha256": PARSER_SHA256,
        "inherited_proof_sha256": PROOF_SHA256, "vertices": {},
    }
    for vertex in ("FFV1L1_2", "FFV1L2_1"):
        dp = HERE/"proof-inputs"/(vertex+".f")
        mp = QP/"proof-inputs/qcd-3"/("MP_"+vertex+".f")
        inherited = proof_data["cases"]["qcd-3"]["vertex_proofs"]["MP_"+vertex]
        assert sha(mp) == inherited["source_sha256"]
        dp_coefficients, dp_text = source_parser.assignments(dp)
        mp_coefficients, _ = source_parser.assignments(mp)
        assert len(dp_coefficients) == len(mp_coefficients) == 80
        assert dp_coefficients.keys() == mp_coefficients.keys()
        for key, value in dp_coefficients.items():
            assert value == mp_coefficients[key], (vertex, key)
        exact = source_parser.prove(dp)
        assert exact["passed"] and exact["coefficient_identities_checked"] == 64
        for field in ("q_rows", "terminal_required_u_rows", "b0_nonzero_by_row",
                      "h_nonzero_by_row"):
            assert exact[field] == inherited[field], (vertex, field)
        logical = source_parser.logical(dp)
        assert "COMPLEX*16CI" in logical
        assert any(line.startswith("REAL*8M") for line in logical)
        assert all("Q" not in text for text in dp_text.values())
        original = ROOT/".benchmark-work/policy-v2/generation-abi3-terminal-fixed/outputs/candidate/qcd-3/Source/DHELAS"/(vertex+".f")
        if original.exists():
            assert original.read_bytes() == dp.read_bytes()
        result["vertices"][vertex] = {
            "source": str(dp.relative_to(ROOT)), "source_sha256": sha(dp),
            "copied_from": str(original.relative_to(ROOT)),
            "inherited_source": str(mp.relative_to(ROOT)),
            "inherited_source_sha256": sha(mp),
            "all_coefficient_source_identities_checked": 80,
            **exact,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(bind(), indent=2, sort_keys=True)+"\n"
    if args.output:
        with args.output.open("x") as target:
            target.write(text)
    else:
        print(text, end="")
