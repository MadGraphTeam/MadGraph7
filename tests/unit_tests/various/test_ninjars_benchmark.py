"""Acceptance controls for the retained numerical/performance benchmark."""
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "ninjars_benchmark", ROOT / "tools/ninjars_benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkAcceptanceTest(unittest.TestCase):
    def test_common_timing_noise_preserves_paired_ratio(self):
        reference = [1., 1.5, .8, 1.2, 1.8, .9, 1.4]
        fast = [x * .9 for x in reference]
        result = benchmark.runtime_statistics(reference, fast, 1.05)
        self.assertEqual(result["status"], "passed")
        for endpoint in result["ratio_confidence_interval_95"]:
            self.assertAlmostEqual(endpoint, .9)
        slow = benchmark.runtime_statistics(reference, [x*1.2 for x in reference], 1.05)
        self.assertEqual(slow["status"], "failed")

    def test_uncertain_or_insufficient_measurement_cannot_pass(self):
        result = benchmark.runtime_statistics([1.]*7, [.9, .95, 1., 1.03, 1.1, 1.2, 1.3], 1.05)
        self.assertLess(result["runtime_ratio"], 1.05)
        self.assertEqual(result["status"], "unresolved")
        self.assertFalse(result["passed"])
        with self.assertRaises(RuntimeError):
            benchmark.runtime_statistics([1.]*6, [.8]*6, 1.05)

    def test_fallback_cannot_count_as_requested_backend(self):
        for backend, codes in {1: (211, 319), 6: (216, 318), 7: (217,),
                               8: (218, 312)}.items():
            for code in codes:
                benchmark.validate_status(code, backend)
        for code in (216, 318, 212, 210, 238, 418):
            with self.assertRaises(RuntimeError, msg=str(code)):
                benchmark.validate_status(code, 8)


if __name__ == "__main__":
    unittest.main()
