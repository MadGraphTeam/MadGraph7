from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from tirust_benchmark_pole_timing import parse_results


class TestPairedTensorTiming(unittest.TestCase):
    def fixture(self, scope="generated_adapter"):
        return dict(scope=scope, ctmode=1, n=1, rank=0, pairs=7,
                    target_seconds=1.1, max_loops=10000, max_calibrations=4)

    def output(self, scope=0):
        lines = [f"H {scope} 1 1 0 1 1.1 7 10000"]
        def frame(phase, repetition, finite, loops=1, seconds=1.1):
            lines.append(f"F {phase} {repetition} {finite} {loops} T 1 {seconds} {seconds}")
            value = "8" if finite else "NaN" if scope == 0 else "0"
            lines.append(f"C {phase} {repetition} {finite} 0 {value} 0 2 0 1 0")
        for finite in (1, 0):
            frame(0, 1, finite)
        for finite in (1, 0):
            frame(1, 1, finite, 10 if finite else 20)
        for pair in range(1, 8):
            for finite in ((1, 0) if pair % 2 else (0, 1)):
                frame(2, pair, finite, 10 if finite else 20)
        return "\n".join(lines + ["E 7"]) + "\n"

    def test_paired_per_call_ratio_and_interval(self):
        result = parse_results(self.output(), self.fixture())
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["wall_poles_over_full"]["geometric_mean"], 0.5)
        self.assertEqual(result["wall_poles_over_full"]["confidence_interval_95"], [0.5, 0.5])
        self.assertFalse(result["performance_eligible"])

    def test_bridge_unevaluated_zero_is_distinct_from_adapter_nan(self):
        result = parse_results(self.output(1), self.fixture("public_fortran_bridge"))
        self.assertEqual(result["status"], "measured")
        bad = self.output().replace("NaN", "0")
        self.assertEqual(parse_results(bad, self.fixture())["status"], "coefficient_failure")

    def test_underlength_measurements_remain_unqualified(self):
        text = self.output().replace("F 2 1 1 10 T 1 1.1", "F 2 1 1 10 T 1 0.1")
        result = parse_results(text, self.fixture())
        self.assertEqual(result["status"], "timing_incomplete")
        self.assertFalse(result["timing_qualified"])

    def test_output_parity_and_fixed_loop_counts_are_checked(self):
        text = self.output().replace("C 2 7 0 0 NaN 0 2", "C 2 7 0 0 NaN 0 3")
        self.assertEqual(parse_results(text, self.fixture())["status"], "coefficient_failure")
        text = self.output().replace("F 2 7 1 10", "F 2 7 1 11")
        with self.assertRaisesRegex(ValueError, "differs from accepted calibration"):
            parse_results(text, self.fixture())

    def test_refusal_has_no_speed_claim(self):
        text = "H 0 1 1 0 1 1.1 7 10000\nF 0 1 1 0 F 1 .001 .001\nF 0 1 0 0 F 1 .001 .001\nE 0\n"
        result = parse_results(text, self.fixture())
        self.assertEqual(result["status"], "refused")
        self.assertNotIn("wall_poles_over_full", result)

    def test_calibration_footer_and_bounds_are_required(self):
        lines = self.output().splitlines()
        no_calibration = "\n".join(line for line in lines if not line.startswith(("F 1 ", "C 1 ")))
        with self.assertRaisesRegex(ValueError, "lack the required calibration"):
            parse_results(no_calibration, self.fixture())
        with self.assertRaisesRegex(ValueError, "follows the completion footer"):
            parse_results(self.output() + "E 7\n", self.fixture())
        excessive = self.output().replace("F 2 1 1 10", "F 2 1 1 10001")
        with self.assertRaisesRegex(ValueError, "exceeds the declared bound"):
            parse_results(excessive, self.fixture())


if __name__ == "__main__":
    unittest.main()
