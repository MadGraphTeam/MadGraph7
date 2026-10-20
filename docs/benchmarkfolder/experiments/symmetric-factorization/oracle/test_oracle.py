#!/usr/bin/env python3
"""Small exact-arithmetic checks of the independent oracle itself."""
import sys
sys.dont_write_bytecode = True

import copy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import unittest

import driver
import oracle


def decode(bits):
    return oracle.binary128(f"{bits >> 64:016x}", f"{bits & (2**64-1):016x}")


class ExactOracleTests(unittest.TestCase):
    def test_ieee_binary128_decode_boundaries(self):
        self.assertEqual(decode(0), (F(0), 0))
        self.assertEqual(decode(1 << 127), (F(0), 1 << 127))
        self.assertEqual(decode(16383 << 112)[0], F(1))
        self.assertEqual(decode((16383 << 112)+1)[0], F(1)+F(1, 2**112))
        self.assertEqual(decode((1 << 127)+(16383 << 112))[0], F(-1))
        self.assertEqual(decode(1)[0], F(1, 2**16494))
        self.assertEqual(decode(1 << 112)[0], F(1, 2**16382))
        self.assertEqual(decode((1 << 112)-1)[0], F((1 << 112)-1, 2**16494))
        self.assertEqual(decode((32766 << 112)+(1 << 112)-1)[0],
                         F((1 << 113)-1) * 2**16271)
        for bits in (32767 << 112, (32767 << 112)+1,
                     (1 << 127)+(32767 << 112)):
            with self.assertRaises(ValueError):
                decode(bits)
        with self.assertRaises(AssertionError):
            oracle.binary128("10000000000000000", "0000000000000000")

    def test_complex_exact_product_and_scale(self):
        a, b = (F(1), F(2)), (F(3), F(-4))
        self.assertEqual(oracle.multiply(a, b), (F(11), F(2)))
        self.assertEqual(oracle.scale_products(a, b), 21)

    def test_monomial_order_and_shift_positions(self):
        known = {(): 0, (0,): 1, (1,): 2, (2,): 3, (3,): 4,
                 (0, 0): 5, (0, 1): 6, (1, 1): 7, (0, 2): 8,
                 (1, 2): 9, (2, 2): 10, (0, 3): 11, (1, 3): 12,
                 (2, 3): 13, (3, 3): 14,
                 (0, 0, 1): 16, (0, 1, 1): 17,
                 (0, 1, 2): 20, (0, 1, 3): 26}
        for monomial, expected in known.items():
            self.assertEqual(oracle.position(monomial), expected)
        for rank, count in ((0, 1), (1, 5), (2, 15), (3, 35), (4, 70), (5, 126)):
            self.assertEqual(len(oracle.monomials(rank)), count)

    def test_sparse_known_contraction_and_deliberate_output_corruption(self):
        case = dict(number=1, rank=2, terminal=False, diagonal_only=False,
                    cut=4, ins=4, outs=4, arrays={}, bits={},
                    immutable={"OLD": (1, 1), "NEW": (1, 1)},
                    tail={"OLD": 1, "NEW": 1},
                    flags={"OLD": (0, 0, 0, 0, 1), "NEW": (0, 0, 0, 0, 1)},
                    trace=(1, 0))
        for name, keys in oracle.expected_arrays(case).items():
            case["arrays"][name] = {key: oracle.ZERO for key in keys}
            case["bits"][name] = {key: (0, 0) for key in keys}
        case["arrays"]["A"][1, 6, 1] = (F(1), F(2))
        case["arrays"]["B"][1, 0, 1] = (F(2), F(0))
        for m in range(1, 5):
            case["arrays"]["B"][1, m, 1] = (F(3), F(-4))
        for name in ("OLD", "NEW"):
            case["arrays"][name][1, 6, 1] = (F(2), F(4))
            for n in (16, 17, 20, 26):
                case["arrays"][name][1, n, 1] = (F(11), F(2))
        rows = [{"source_row": j, "phase": [1, 0]} for j in range(1, 5)]
        proof = {"q_rows": [copy.deepcopy(rows) for _ in range(4)]}
        checked = oracle.check_case(case, proof)
        self.assertTrue(checked["passed"])
        self.assertEqual(checked["max_error_over_epsilon_product_scale"],
                         {"OLD": 0.0, "NEW": 0.0})
        corrupted = copy.deepcopy(case)
        corrupted["arrays"]["NEW"][1, 20, 1] = (F(12), F(2))
        with self.assertRaises(AssertionError):
            oracle.check_case(corrupted, proof)

    def test_generated_sources_and_provenance(self):
        self.assertEqual(hashlib.sha256(driver.PROOF.read_bytes()).hexdigest(),
                         oracle.PROOF_SHA256)
        proofs = json.loads(driver.PROOF.read_text())["cases"]["qcd-4"]["vertex_proofs"]
        passed = [name for name, proof in proofs.items() if proof["passed"]]
        self.assertEqual(len(passed), 10)
        for vertex in passed:
            for rank in (2, 3, 4):
                for terminal in (False, True):
                    source = driver.driver(rank, vertex, terminal)
                    self.assertLessEqual(max(map(len, source.splitlines())), 132)
                    self.assertIn("do trial=1,10", source)
                    self.assertIn("PASS_DUMP", source)
                    self.assertIn("saved_a=transfer(A,saved_a)", source)
                    self.assertIn("saved_b=transfer(B,saved_b)", source)
                    self.assertIn("call alter_zero_sign", source)


if __name__ == "__main__":
    unittest.main()
