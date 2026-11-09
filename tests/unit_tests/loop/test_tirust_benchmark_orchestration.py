"""Keep benchmark configuration local and defer costly unqualified campaigns."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as benchmark


class TestBenchmarkCLI(unittest.TestCase):
    def test_every_configuration_command_is_session_local(self):
        case = {"model": "loop_sm", "process": "g g > h [noborn=QCD]"}
        with patch.object(Path, "is_dir", return_value=True):
            for source in ("baseline", "candidate"):
                lines = benchmark.cli_lines(ROOT, source, case, Path("/private/output"))
                sets = [line for line in lines if line.startswith("set ")]
                self.assertGreater(len(sets), 6)
                self.assertTrue(all(line.split()[-1] == "--no_save" for line in sets))
                self.assertTrue(all(line.count("--no_save") == 1 for line in sets))
                self.assertEqual(any(line.startswith("set tirust ") for line in sets), source == "candidate")
                self.assertIn("generate g g > h [sqrvirt=QCD]", lines)
                self.assertEqual(lines[-1], "quit")
                self.assertFalse(any("--no_save" in line for line in lines if not line.startswith("set ")))


class TestBenchmarkCampaignOrder(unittest.TestCase):
    def setUp(self):
        self.cases = [{"id": "first", "role": "main"},
                      {"id": "control", "role": "control"},
                      {"id": "second", "role": "main"}]
        self.calls, self.failures = [], []

    def run_task(self, case, backend):
        self.calls.append((case["id"], backend))
        return {"acceptance_eligible": True, "passed": True}

    def test_all_main_chain_gates_precede_any_explicit_regression(self):
        benchmark.execute_ordered_campaign(self.cases, self.run_task, self.failures.append)
        self.assertEqual(self.calls[:2], [("first", None), ("second", None)])
        self.assertEqual(len(self.calls), 18)
        self.assertEqual(self.calls.count(("control", None)), 1)
        self.assertFalse(self.failures)

    def test_failed_eligible_chain_preserves_other_main_gates_and_defers_rest(self):
        def run(case, backend):
            result = self.run_task(case, backend)
            result["passed"] = case["id"] != "first"
            return result
        with self.assertRaisesRegex(RuntimeError, "regressions and controls are deferred"):
            benchmark.execute_ordered_campaign(self.cases, run, self.failures.append)
        self.assertEqual(self.calls, [("first", None), ("second", None)])
        self.assertEqual(self.failures[0]["case"], "first")

    def test_ineligible_or_incomplete_chain_cannot_open_regression_phase(self):
        for incomplete in (False, True):
            with self.subTest(incomplete=incomplete):
                self.calls.clear()
                self.failures.clear()
                def run(case, backend):
                    result = self.run_task(case, backend)
                    if case["id"] == "first":
                        if incomplete:
                            raise ValueError("retained raw evidence incomplete")
                        result["acceptance_eligible"] = False
                    return result
                with self.assertRaisesRegex(RuntimeError, "gates failed/incomplete"):
                    benchmark.execute_ordered_campaign(self.cases, run, self.failures.append)
                self.assertEqual(self.calls, [("first", None), ("second", None)])
                self.assertEqual(len(self.failures), 1)

    def test_regression_failure_is_retained_without_hiding_remaining_regressions(self):
        def run(case, backend):
            result = self.run_task(case, backend)
            if case["id"] == "first" and backend == 6:
                result["passed"] = False
            return result
        with self.assertRaisesRegex(RuntimeError, "retains 1 failed/incomplete"):
            benchmark.execute_ordered_campaign(self.cases, run, self.failures.append)
        self.assertEqual(len(self.calls), 18)
        self.assertEqual(self.failures[0]["backend"], 6)

    def test_eligible_slow_standalone_tirust_and_controls_are_descriptive(self):
        def run(case, backend):
            result = self.run_task(case, backend)
            if backend == 9 or case["role"] == "control":
                result["passed"] = False
            return result
        benchmark.execute_ordered_campaign(self.cases, run, self.failures.append)
        self.assertEqual(len(self.calls), 18)
        self.assertFalse(self.failures)

    def test_numerical_or_duration_failure_is_fatal_for_descriptive_rows_too(self):
        for failed_row in (("first", 9), ("control", 6), ("control", None)):
            with self.subTest(failed_row=failed_row):
                self.calls.clear()
                self.failures.clear()
                def run(case, backend):
                    result = self.run_task(case, backend)
                    if (case["id"], backend) == failed_row:
                        result.update(acceptance_eligible=False, passed=False)
                    return result
                with self.assertRaisesRegex(RuntimeError, "retains 1 failed/incomplete"):
                    benchmark.execute_ordered_campaign(self.cases, run, self.failures.append)
                self.assertEqual(len(self.calls), 18)
                self.assertEqual(len(self.failures), 1)


if __name__ == "__main__":
    unittest.main()
