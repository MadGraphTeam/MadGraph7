"""Acceptance gates must be based on complete, matching retained observations."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as benchmark


class TestTIRustBenchmark(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.report_patch = patch.object(benchmark, "REPORT", self.directory)
        self.report_patch.start()

    def tearDown(self):
        self.report_patch.stop()
        self.temporary.cleanup()

    def config(self):
        return json.loads((ROOT / "docs/benchmarkfolder/cases.json").read_text())

    def test_matrix_covers_requested_multiplicities(self):
        config = self.config()
        main = [case for case in config["cases"] if case["role"] == "main"]
        self.assertEqual(len(main), 8)
        for model in ("loop_sm", "loop_qcd_qed_sm_Gmu"):
            self.assertEqual(sorted(case["nfinal"] for case in main if case["model"] == model), [1, 2, 3, 4])
        self.assertEqual(config["points_per_case"], 100)
        self.assertEqual(config["baseline_revision"], benchmark.BASELINE_REVISION)
        self.assertEqual(config["backends"], [1, 6, 7, 8, 9])

    def test_missing_data_stays_pending(self):
        summary = benchmark.summarize(self.config())
        self.assertFalse(summary["gate"]["complete"])
        self.assertFalse(summary["gate"]["passed"])
        self.assertEqual(len(summary["gate"]["pending_cases"]), 8)
        for case in summary["cases"]:
            self.assertIsNone(case["performance"]["geometric_mean_ratio"])
            self.assertIsNone(case["points"])
            for timing in case["timings"]:
                self.assertIsNone(timing["median_seconds_per_point"])
                self.assertEqual(timing["samples"], 0)

    def test_frozen_points_reject_changed_mass_and_file(self):
        config = self.config()
        case = config["cases"][0]
        filename = benchmark.freeze_points(case, [0., 0., 91.188], config)
        self.assertEqual(int(filename.read_text().splitlines()[0]), 100)
        self.assertEqual(benchmark.freeze_points(case, [0., 0., 91.188], config), filename)
        with self.assertRaisesRegex(RuntimeError, "Frozen"):
            benchmark.freeze_points(case, [0., 0., 91.2], config)
        filename.write_text(filename.read_text().replace("100\n", "99\n", 1))
        with self.assertRaisesRegex(RuntimeError, "Frozen"):
            benchmark.freeze_points(case, [0., 0., 91.188], config)

    def test_laurent_mismatch_and_reordered_points_fail(self):
        left = [{"point": 1, "order": 0, "coefficients": [1., 2., -3., 4.]}]
        right = [{"point": 1, "order": 0, "coefficients": [1., 2., -3.1, 4.]}]
        result = benchmark.compare_rows(left, right)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"][0]["column"], 2)
        right[0]["point"] = 2
        with self.assertRaisesRegex(RuntimeError, "identical"):
            benchmark.compare_rows(left, right)

    def test_backend_aliases_and_chain_validation(self):
        self.assertEqual(benchmark.parse_chain("9|8|7|6|1"), [9, 8, 7, 6, 1])
        for chain in ("", "9|9", "0", "10", "auto"):
            with self.assertRaises(ValueError):
                benchmark.parse_chain(chain)

    def test_exact_comparator_small_poles_and_opposite_signs(self):
        def rows(values):
            return [{"point": 1, "order": 0, "coefficients": values}]
        # Opposite tiny poles use the finite-term exception, not the large Born.
        left = rows([1e9, 1., 1e-7, 0.])
        right = rows([1e9, 1., -1e-7, 0.])
        result = benchmark.compare_rows(left, right)
        self.assertTrue(result["passed"])
        self.assertFalse(result["ordinary_target_passed"])
        self.assertIsNone(result["residuals"][2]["symmetric_relative_error"])
        self.assertFalse(benchmark.compare_rows(rows([1e9, 1e-6, .1, 0.]),
                                               rows([1e9, 1e-6, -.1, 0.]))["passed"])
        self.assertFalse(benchmark.compare_rows(rows([0., 1., 0., 0.]),
                                               rows([0., -1., 0., 0.]))["passed"])
        self.assertTrue(benchmark.compare_rows(rows([0., 0., 0., 0.]),
                                              rows([0., 0., 0., 0.]))["passed"])

    def test_startup_failures_never_complete_campaign(self):
        config = self.config()
        for case in config["cases"]:
            benchmark.dump(self.directory / "runs" / case["id"] / "tirust-first/failure.json",
                           {"status": "failed", "error": "synthetic startup failure"})
        summary = benchmark.summarize(config)
        self.assertFalse(summary["gate"]["complete"])
        self.assertFalse(summary["gate"]["passed"])
        self.assertEqual(len(summary["gate"]["failed_cases"]), 8)

    def test_unbound_summary_is_failed_not_measured(self):
        config = self.config()
        case = config["cases"][0]
        benchmark.dump(self.directory / "runs" / case["id"] / "tirust-first/summary.json",
                       {"status": "complete", "passed": True})
        summary = benchmark.summarize(config)
        row = summary["cases"][0]
        self.assertEqual(row["status"], "failed")
        self.assertFalse(row["complete"])
        self.assertIn("tirust-first", row["validation_errors"])
        self.assertIsNone(row["performance"]["geometric_mean_ratio"])

    def test_failure_marker_dominates_a_passing_summary(self):
        config = self.config()
        case = config["cases"][0]
        directory = self.directory / "run"
        benchmark.dump(directory / "summary.json", {"status": "complete", "passed": True})
        benchmark.dump(directory / "failure.json", {"error": "worker shutdown failed"})
        with self.assertRaisesRegex(ValueError, "retained failure"):
            benchmark.validate_retained_result(directory, "tirust-first", case, config)


if __name__ == "__main__":
    unittest.main()
