"""Keep benchmark configuration local and defer costly unqualified campaigns."""
from pathlib import Path
import copy
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as benchmark
import tirust_benchmark_numerical as numerical
import tirust_benchmark_campaign as campaign


class TestExportEnvironment(unittest.TestCase):
    def test_child_environment_overrides_inherited_hash_randomization(self):
        for inherited in ("random", "123456"):
            with self.subTest(inherited=inherited), patch.dict(os.environ, {"PYTHONHASHSEED": inherited}):
                self.assertEqual(benchmark.environment()["PYTHONHASHSEED"], "0")
                self.assertEqual(os.environ["PYTHONHASHSEED"], inherited)

    def test_fresh_generation_requires_complete_exact_binding(self):
        good = {"source": "candidate", **campaign.export_environment_binding()}
        campaign.validate_generation_environment(good, required=True)
        for field in campaign.export_environment_binding():
            changed = dict(good)
            changed.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                campaign.validate_generation_environment(changed, required=True)
        for bad in (None, {}, {"PYTHONHASHSEED": 0}, {"PYTHONHASHSEED": "random"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                campaign.validate_generation_environment(dict(good, export_environment=bad), required=True)
        campaign.validate_generation_environment({"source": "baseline"})
        with self.assertRaises(ValueError):
            campaign.validate_generation_environment({"source": "candidate"}, required=True)

    def test_cached_new_campaign_generation_rejects_missing_or_changed_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            binding = {"id": "new", "manifest_sha256": "frozen"}
            record = {"source": "candidate", "campaign_binding": binding,
                      **campaign.export_environment_binding()}
            with patch.object(benchmark, "output_dir", return_value=output), \
                 patch.object(benchmark, "campaign_guard"), \
                 patch.object(benchmark, "campaign_binding", return_value=binding), \
                 patch.object(benchmark, "CAMPAIGN_ID", "new"):
                (output / "generation.json").write_text(json.dumps(record))
                self.assertEqual(benchmark.generate("candidate", {"id": "example"}), record)
                for bad in (None, {"PYTHONHASHSEED": "1"}):
                    changed = dict(record)
                    if bad is None:
                        for key in campaign.export_environment_binding():
                            changed.pop(key)
                    else:
                        changed["export_environment"] = bad
                    (output / "generation.json").write_text(json.dumps(changed))
                    with self.subTest(bad=bad), self.assertRaises(ValueError):
                        benchmark.generate("candidate", {"id": "example"})

    def test_inherited_baseline_keeps_unrecorded_export_seed_separate_from_relink(self):
        reused = {"source": "baseline", "reused_from": {
            "source_generation_repeated": False,
            "source_export_environment_recorded": False,
            "source_export_environment": None,
            "reuse_environment_binding": campaign.export_environment_binding()}}
        campaign.validate_generation_environment(reused, required=True)
        self.assertNotIn("export_environment", reused)
        changes = [dict(reused, source="candidate"),
                   dict(reused, **campaign.export_environment_binding())]
        for field, value in (("source_generation_repeated", True),
                             ("source_export_environment_recorded", True),
                             ("source_export_environment", {"PYTHONHASHSEED": "0"}),
                             ("reuse_environment_binding", None)):
            changed = copy.deepcopy(reused)
            changed["reused_from"][field] = value
            changes.append(changed)
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                campaign.validate_generation_environment(changed, required=True)

    def test_recorded_legacy_binding_cannot_change_even_in_read_only_validation(self):
        historical = {"source": "candidate", **campaign.export_environment_binding()}
        campaign.validate_generation_environment(historical)
        historical["export_environment"] = {"PYTHONHASHSEED": "23"}
        with self.assertRaises(ValueError):
            campaign.validate_generation_environment(historical)


class TestWorkerTimeoutBinding(unittest.TestCase):
    case = {"id": "example"}

    def test_manifest_per_case_value_and_legacy_default(self):
        with patch.object(benchmark, "CAMPAIGN_ID", None):
            self.assertEqual(benchmark.case_worker_timeout(self.case), 3600.)
        with patch.object(benchmark, "CAMPAIGN_ID", "new"), \
             patch.object(benchmark, "CAMPAIGN_MANIFEST", {
                 "administrative_worker_timeout_seconds": {"example": 14400}}):
            self.assertEqual(benchmark.case_worker_timeout(self.case), 14400.)
        for value in (None, {}, {"example": True}, {"example": 0}, {"example": float("nan")}):
            with self.subTest(value=value), patch.object(benchmark, "CAMPAIGN_ID", "new"), \
                 patch.object(benchmark, "CAMPAIGN_MANIFEST", {"administrative_worker_timeout_seconds": value}):
                with self.assertRaises(ValueError):
                    benchmark.case_worker_timeout(self.case)

    def test_new_protocol_rejects_changed_and_missing_spec_or_worker_bindings(self):
        spec = {"worker_timeout_binding_version": 1, "worker_timeout_seconds": 3600.}
        inputs = {role: {"worker_timeout_seconds": 3600.} for role in ("baseline", "candidate")}
        path = ROOT / "tools/tirust_benchmark.py"
        benchmark.validate_worker_timeout_binding(spec, inputs, self.case, path)
        for field in ("worker_timeout_binding_version", "worker_timeout_seconds"):
            changed = copy.deepcopy(spec)
            changed.pop(field)
            with self.subTest(missing=field), self.assertRaises(ValueError):
                benchmark.validate_worker_timeout_binding(changed, inputs, self.case, path)
        with self.assertRaises(ValueError):
            benchmark.validate_worker_timeout_binding({}, inputs, self.case, path)
        with self.assertRaises(ValueError):
            benchmark.validate_worker_timeout_binding(dict(spec, worker_timeout_binding_version=True), inputs, self.case, path)
        for role in inputs:
            for bad in (None, 14400.):
                changed = copy.deepcopy(inputs)
                if bad is None:
                    changed[role].pop("worker_timeout_seconds")
                else:
                    changed[role]["worker_timeout_seconds"] = bad
                with self.subTest(role=role, bad=bad), self.assertRaises(ValueError):
                    benchmark.validate_worker_timeout_binding(spec, changed, self.case, path)
        with self.assertRaises(ValueError):
            benchmark.validate_worker_timeout_binding(dict(spec, worker_timeout_seconds=14400.), inputs, self.case, path)

    def test_pre_binding_historical_protocol_is_readable_without_new_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical.py"
            path.write_text("HISTORICAL_PROTOCOL = 1\n")
            with patch.object(benchmark, "CAMPAIGN_ID", None):
                benchmark.validate_worker_timeout_binding({}, {"baseline": {}, "candidate": {}}, self.case, path)
            with patch.object(benchmark, "CAMPAIGN_ID", "new"):
                with self.assertRaises(ValueError):
                    benchmark.validate_worker_timeout_binding({}, {"baseline": {}, "candidate": {}}, self.case, path)

    def test_numerical_preparation_uses_and_enforces_preregistered_deadline(self):
        with patch.object(benchmark, "CAMPAIGN_ID", "new"), \
             patch.object(benchmark, "CAMPAIGN_MANIFEST", {
                 "administrative_worker_timeout_seconds": {"example": 14400.}}):
            self.assertEqual(numerical.resolve_worker_timeout(self.case), 14400.)
            self.assertEqual(numerical.resolve_worker_timeout(self.case, 14400.), 14400.)
            with self.assertRaisesRegex(ValueError, "preregistered"):
                numerical.resolve_worker_timeout(self.case, 3600.)
        with patch.object(benchmark, "CAMPAIGN_ID", None):
            self.assertEqual(numerical.resolve_worker_timeout(self.case), 3600.)
            self.assertEqual(numerical.resolve_worker_timeout(self.case, 1800.), 1800.)
            for invalid in (True, 0, -1, float("inf"), float("nan")):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    numerical.resolve_worker_timeout(self.case, invalid)


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
