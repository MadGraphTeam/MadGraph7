#!/usr/bin/env python3
"""Deferred exact-arithmetic self-checks; execute after quiet-window release."""
import sys
sys.dont_write_bytecode = True

import copy
from fractions import Fraction as F
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import driver
import oracle


def decode(bits):
    return oracle.binary64(f"{bits:016x}")


def known_case():
    case = dict(number=1, rank=2, terminal=False, diagonal_only=False,
                cut=4, ins=4, outs=4, arrays={}, bits={},
                immutable={"OLD":(1,1), "NEW":(1,1)},
                tail={"OLD":1, "NEW":1},
                flags={"OLD":(0,0,0,0,1), "NEW":(0,0,0,0,1)}, trace=(1,0))
    for name, keys in oracle.expected_arrays(case).items():
        case["arrays"][name] = {key:oracle.ZERO for key in keys}
        case["bits"][name] = {key:(0,0) for key in keys}
    encodings = {0:0, 1:0x3ff0000000000000, 2:0x4000000000000000,
                 3:0x4008000000000000, 4:0x4010000000000000,
                 -4:0xc010000000000000, 11:0x4026000000000000}
    def assign(name, key, real, imag):
        case["arrays"][name][key] = (F(real), F(imag))
        case["bits"][name][key] = (encodings[real], encodings[imag])
    # A occupies two cut components, so terminal projection is nontrivial.
    for i in (1, 2):
        assign("A", (1,6,i), 1, 2)
        for name in ("OLD", "NEW"):
            assign(name, (1,6,i), 2, 4)
            for n in (16,17,20,26):
                assign(name, (1,n,i), 11, 2)
    assign("B", (1,0,1), 2, 0)
    for m in range(1,5):
        assign("B", (1,m,1), 3, -4)
    rows = [{"source_row":j, "phase":[1,0]} for j in range(1,5)]
    return case, {"q_rows":[copy.deepcopy(rows) for _ in range(4)]}


class ExactBinary64OracleTests(unittest.TestCase):
    def test_ieee_decode_boundaries_and_one_ulp(self):
        self.assertEqual(decode(0), (F(0), 0))
        self.assertEqual(decode(1 << 63), (F(0), 1 << 63))
        self.assertEqual(decode(1023 << 52)[0], F(1))
        self.assertEqual(decode((1023 << 52)+1)[0], F(1)+F(1, 2**52))
        self.assertNotEqual(decode(1023 << 52)[0], decode((1023 << 52)+1)[0])
        self.assertEqual(decode((1 << 63)+(1023 << 52))[0], F(-1))
        self.assertEqual(decode(1)[0], F(1, 2**1074))
        self.assertEqual(decode(1 << 52)[0], F(1, 2**1022))
        self.assertEqual(decode((1 << 52)-1)[0], F((1 << 52)-1, 2**1074))
        self.assertEqual(decode((2046 << 52)+(1 << 52)-1)[0],
                         F((1 << 53)-1)*2**971)
        for bits in (2047 << 52, (2047 << 52)+1, (1 << 63)+(2047 << 52)):
            with self.assertRaises(ValueError):
                decode(bits)
        with self.assertRaises(AssertionError):
            oracle.binary64("10000000000000000")

    def test_exponent_endpoints_and_exact_lattice(self):
        admitted = [0, 1 << 63]
        for sign in (0, 1 << 63):
            for exponent in (823, 824, 1023, 1222, 1223):
                for mantissa in (0, 1, (1 << 52)-1):
                    bits = sign | (exponent << 52) | mantissa
                    admitted.append(bits)
                    value = decode(bits)[0]
                    self.assertEqual((value*2**252).denominator, 1)
                    self.assertLess(abs(value), 2**201)
        for bits in admitted:
            self.assertTrue(oracle.admitted_bits(bits))
        for bits in (1, (1 << 52)-1, 822 << 52, 1224 << 52, 2047 << 52):
            self.assertFalse(oracle.admitted_bits(bits))
            self.assertFalse(oracle.admitted_bits(bits | (1 << 63)))
        for left in admitted:
            for right in admitted:
                product = decode(left)[0]*decode(right)[0]
                self.assertEqual((product*2**504).denominator, 1)
                self.assertLess(abs(product), 2**402)

    def test_complex_arithmetic_and_monomial_indices(self):
        self.assertEqual(oracle.multiply((F(1), F(2)), (F(3), F(-4))), (F(11), F(2)))
        self.assertEqual(oracle.scale_products((F(1), F(2)), (F(3), F(-4))), 21)
        known = {():0, (0,):1, (1,):2, (2,):3, (3,):4,
                 (0,0):5, (0,1):6, (1,1):7, (0,2):8, (1,2):9,
                 (2,2):10, (0,3):11, (1,3):12, (2,3):13, (3,3):14,
                 (0,0,1):16, (0,1,1):17, (0,1,2):20, (0,1,3):26}
        for monomial, index in known.items():
            self.assertEqual(oracle.position(monomial), index)
        for rank, size in ((0,1), (1,5), (2,15), (3,35), (4,70), (5,126)):
            self.assertEqual(len(oracle.monomials(rank)), size)

    def test_known_direct_contraction_and_corruption(self):
        case, proof = known_case()
        self.assertTrue(oracle.check_case(case, proof)["passed"])
        case["arrays"]["NEW"][1,20,1] = (F(12), F(2))
        with self.assertRaises(AssertionError):
            oracle.check_case(case, proof)

    def test_future_terminal_uses_full_reference_and_fallback(self):
        case, proof = known_case()
        case.update(terminal=True, diagonal_only=True)
        with self.assertRaises(AssertionError):
            oracle.check_case(case, proof)  # Full NEW does not satisfy projection.
        for key in case["arrays"]["NEW"]:
            if key[0] != key[2]:
                case["arrays"]["NEW"][key] = oracle.ZERO
                case["bits"]["NEW"][key] = (0,0)
        self.assertTrue(oracle.check_case(case, proof)["passed"])
        self.assertNotEqual(case["arrays"]["OLD"][1,16,2], oracle.ZERO)
        case.update(number=8, diagonal_only=False, trace=(0,1))
        with self.assertRaises(AssertionError):
            oracle.check_case(case, proof)  # Fallback must retain full output.
        case["arrays"]["NEW"] = copy.deepcopy(case["arrays"]["OLD"])
        case["bits"]["NEW"] = copy.deepcopy(case["bits"]["OLD"])
        self.assertTrue(oracle.check_case(case, proof)["passed"])

    def test_incomplete_stream_fails(self):
        text = "FORMAT BINARY64_HEX_V1\nLAYOUT 64 53 1024\nPASS_DUMP\n"
        with patch("builtins.open", return_value=io.StringIO(text)):
            with self.assertRaises(AssertionError):
                oracle.read_cases(Path("unopened-stream.txt"))

    def test_actual_dp_sources_and_all_requested_driver_shapes(self):
        data = json.loads(driver.PROOF.read_text())
        self.assertTrue(data["complete"] and data["passed"])
        self.assertEqual(set(data["vertices"]), {"FFV1L1_2", "FFV1L2_1"})
        for vertex in data["vertices"]:
            for rank in (2,3,4):
                for terminal in (False,True):
                    source = driver.driver(rank, vertex, terminal)
                    self.assertLessEqual(max(map(len,source.splitlines())), 132)
                    self.assertIn("complex*16", source)
                    self.assertIn("FORMAT BINARY64_HEX_V1", source)
                    self.assertIn("LAYOUT',64,53,1024", source)
                    self.assertIn("do trial=1,10", source)
                    self.assertIn("PASS_DUMP", source)
                    self.assertIn(f"call TEST_UPDATE_WL_{rank}_1(A,4,B,4,4,oldout)", source)
                    self.assertNotIn(f"call TEST_TERMINAL_WL_{rank}_1(", source)
                    self.assertNotIn("_16", source)
                    self.assertNotIn("Q0", source)


if __name__ == "__main__":
    unittest.main()
