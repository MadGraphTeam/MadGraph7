"""NRS report authority and presentation contracts; synthetic data, no physics."""
import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import ninjars_benchmark_report_v3 as report


class NinjaRSReportV3Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=report.ROOT / ".ninjars-build/tmp", prefix="report-v3-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = {}
        self.config = json.loads((report.ROOT / "docs/benchmarkfolder/policy-v2/cases.json").read_text())

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        bound = report.v2.bound_file(path)
        self.inputs[bound["path"]] = bound["sha256"]
        return bound

    def record(self, cid, backend, repetition, offset=0.):
        code = {1: 211, 6: 216, 8: 218}[backend]
        source = "candidate" if backend == 8 else "baseline"
        entries = [{"point": p, "order": order, "status": code,
                    "coefficients": [float(p), 2.*p + offset, 3.*p, 4.*p], "accuracy": 1e-12}
                   for p in range(1, 101) for order in (0, 1)]
        seconds = self.ratios[repetition % len(self.ratios)] if backend == 8 and type(repetition) is int else (.9 if backend == 8 else 1.)
        return {"case": cid, "source": source, "backend": backend, "repetition": repetition,
                "rows": entries, "status_counts": {str(code): 100},
                "timed_status_counts": {str(code): 100}, "loops": 1,
                "answer_dimensions": {"allocated": 1, "loop_orders": 1, "born_orders": 1},
                "seconds": seconds*100, "seconds_per_point": seconds, "cold_seconds": .5,
                "accuracy": 1e-12, "cpu": 4, "executable_sha256": "synthetic",
                "shared_numerical_settings": report.authority.expected_settings(),
                "parameter_card_sha256": "parameters", "points_sha256": "points"}

    @staticmethod
    def downstream():
        return {"runtime": report.pending("Separate runtime proof absent"),
                "qualifications": {k: report.pending("Independent proof absent")
                                   for k in report.authority.QUALIFICATION_KEYS},
                "profiles": [], "profile_status": "pending"}

    def fixture(self, pair=True, offset=0., external=False, ratios=None, baseline_offset=0.):
        self.ratios = ratios or [.9]*7
        config = self.write("configuration.json", self.config)
        correction = self.write("correction.json", {"identity_sha256": "correction"})
        proof = self.write("proof.json", {"entries": []})
        self.rules = {"identity_sha256": "rules", "configuration": config,
                      "correction": correction, "representative_proof": proof}
        policy = self.write("policy.json", self.rules)
        entries, cases = [], {}
        for definition in self.config["cases"]:
            cid = definition["id"]
            refs = {b: self.record(cid, b, "calibration") for b in (1, 6)}
            for b in (1, 6):
                entries.append({"case": cid, "backend": b,
                                "record": self.write(f"references/{cid}-{b}.json", refs[b])})
            checks, timing_checks = [], []
            nrs = self.record(cid, 8, 0, offset)
            groups = {"original_helper_corrected/1": refs[1]["rows"],
                      "original_helper_corrected/6": refs[6]["rows"], "candidate_corrected/8": nrs["rows"]}
            result = report.v2.compare_rows(groups)
            for rep in range(2):
                self.write(f"correct/data/{cid}-candidate-8-{rep}.json", self.record(cid, 8, rep, offset))
                checks.append(copy.deepcopy(result))
            if pair:
                for rep in range(7):
                    for b, source in ((6, "baseline"), (8, "candidate")):
                        self.write(f"pair/data/{cid}-{source}-{b}-{rep}.json", self.record(cid, b, rep, offset if b == 8 else baseline_offset))
                    checks.append(copy.deepcopy(result))
                    timing_groups = dict(groups, **{"original/6": self.record(cid, 6, rep, baseline_offset)["rows"]})
                    timing_checks.append(report.v2.compare_rows(timing_groups))
            performance = report.legacy.bench.runtime_statistics([1.]*7, self.ratios, 1.05, self.config["seed"]) if pair else {"status": "pending", "passed": False}
            cases[cid] = {"agreement": {"status": "passed" if result["passed"] else "failed",
                                       "passed": result["passed"], "comparisons": checks},
                          "performance": performance,
                          "timing_agreement": {"status": "passed" if timing_checks and all(c["passed"] for c in timing_checks) else "failed" if timing_checks else "pending",
                                               "passed": bool(timing_checks) and all(c["passed"] for c in timing_checks),
                                               "comparisons": timing_checks}}
        self.references = {"entries": entries}
        reference_binding = self.write("references.json", self.references)
        correctness = self.write("correct-binding.json", {"report": str(self.root / "correct"), "targets": {}})
        timing = self.write("pair-binding.json", {"report": str(self.root / "pair"), "targets": {}}) if pair else None
        ext = None
        if external:
            failed = self.write("collier-failure.json", {"backend": 7, "stage": "calibration",
                "run_status": "failed", "raw_outputs": {"timed_status.dat": "217 98\n410 2\n"}})
            ext = self.write("external.json", {"schema": "nrs-external-findings-v3", "findings": [
                {"backend": 7, "case": "qcd-4", "status": "failed", "points": [17, 52], "evidence": [failed]}]})
        correctness_passed = not offset
        self.dataset = {"schema": report.authority.DATASET, "identity_sha256": "synthetic-dataset",
            "policy": policy, "reference_bindings": reference_binding,
            "correctness_bindings": correctness, "timing_bindings": timing, "external_findings": ext,
            "cases": cases, "nrs_correctness_passed": correctness_passed,
            "nrs_performance_passed": pair and all(c["performance"]["passed"] and c["timing_agreement"]["passed"] for c in cases.values()),
            "nrs_core_validation_passed": correctness_passed and pair and all(c["performance"]["passed"] and c["timing_agreement"]["passed"] for c in cases.values()),
            "nrs_validation_milestone_passed": True, "final_publication_passed": True,
            "inputs_sha256": dict(self.inputs), "measurements_sha256": {}}
        self.dataset_path = self.root / "dataset.json"
        self.dataset_path.write_text(json.dumps(self.dataset))

    def snapshot(self, **kwargs):
        with ExitStack() as stack:
            stack.enter_context(patch.object(report.authority, "validate_dataset", return_value=self.dataset))
            stack.enter_context(patch.object(report.authority, "validate_policy", return_value=(
                self.rules, {"identity_sha256": "correction"}, self.config, {"entries": []})))
            stack.enter_context(patch.object(report, "qualification_snapshot", return_value=self.downstream()))
            return report.build_snapshot(self.dataset_path, timestamp="synthetic test", **kwargs)

    def completion_view(self):
        binding = self.dataset["correctness_bindings"]
        destination = next(iter(sorted((self.root / "correct/data").glob("*.json"))))
        original = self.write("original-raw.json", json.loads(destination.read_text()))
        receipt = {"schema": "nrs-first-core-correctness-finalization-v1",
            "status": "complete-raw-correctness-only", "bindings": binding,
            "original_status": self.write("original-status.json", {"status": "failed", "error_type": "RuntimeError"}),
            "view_status": self.write("view-status.json", {"status": "complete"}),
            "original_protocol": self.write("original-protocol.json", {"scope": "synthetic original loaded protocol"}),
            "original_specification": self.write("original-spec.json", {"scope": "synthetic original loaded source"}),
            "copies": [{"source": original, "destination": report.v2.bound_file(destination)}],
            "original_failure_promoted": False, "numerical_retries": 0,
            "new_amplitude_evaluations": 0, "fresh_parity": False}
        self.dataset["correctness_bindings"] = self.write("finalization.json", receipt)
        self.dataset["inputs_sha256"] = dict(self.inputs)
        self.dataset_path.write_text(json.dumps(self.dataset))
        return receipt

    def test_derived_completion_view_preserves_original_failure_and_no_fresh_claim(self):
        self.fixture(pair=False)
        self.completion_view()
        result = self.snapshot()
        provenance = result["correctness_collection_provenance"]
        self.assertTrue(result["nrs_correctness_passed"])
        self.assertEqual(provenance["kind"], "derived-raw-correctness-copy-view")
        self.assertEqual(provenance["original_status"]["status"], "failed")
        self.assertEqual(provenance["view_status"]["status"], "complete")
        self.assertFalse(provenance["original_controller_success"])
        self.assertFalse(provenance["fresh_process_parity_passed"])
        self.assertEqual(provenance["new_amplitude_evaluations"], 0)

    def test_derived_view_cannot_promote_original_controller_failure(self):
        self.fixture(pair=False)
        receipt = self.completion_view()
        receipt["original_failure_promoted"] = True
        self.dataset["correctness_bindings"] = self.write("finalization.json", receipt)
        self.dataset["inputs_sha256"] = dict(self.inputs)
        self.dataset_path.write_text(json.dumps(self.dataset))
        with self.assertRaisesRegex(ValueError, "cannot promote"):
            self.snapshot()

    def test_external_410_failure_stays_visible_without_vetoing_nrs_core(self):
        self.fixture(external=True)
        result = self.snapshot()
        self.assertTrue(result["nrs_core_validation_passed"])
        self.assertEqual(result["external_backend_comparison_status"], "failed")
        self.assertEqual(result["external_diagnostics"]["findings"][0]["points"], [17, 52])
        self.assertFalse(result["external_diagnostics"]["vetoes_nrs_core"])
        self.assertFalse(result["external_backend_comparison_passed"])
        self.assertEqual(next(c for c in result["cases"] if c["id"] == "qcd-4")["external_backend_cells"]["7"], "failed")

    def test_core_pass_cannot_self_certify_qualification_or_publication(self):
        self.fixture()
        result = self.snapshot()
        self.assertTrue(result["nrs_correctness_passed"])
        self.assertTrue(result["nrs_performance_passed"])
        self.assertFalse(result["nrs_full_qualification_passed"])
        self.assertFalse(result["final_publication_passed"])
        self.assertEqual(result["publication_obligations"]["profiles"]["status"], "pending")
        self.assertIsNone(result["external_failures_recorded"])

    def test_two_correctness_commands_do_not_become_performance_samples(self):
        self.fixture(pair=False)
        result = self.snapshot()
        self.assertTrue(result["nrs_correctness_passed"])
        self.assertFalse(result["nrs_performance_passed"])
        self.assertFalse(result["nrs_core_validation_passed"])
        case = result["cases"][0]
        self.assertEqual(case["nrs_numerical_commands"], 2)
        self.assertEqual(case["stability"]["selected_observation"]["repetition"], 0)
        self.assertEqual(case["timings"]["candidate"]["8"]["repetitions"], 0)
        for ref in case["numerical_reference_usage"].values():
            self.assertEqual(ref["timing_repetitions"], 0)
            self.assertIsNone(ref["median_us"])
            self.assertEqual(ref["physical_points"], 100)

    def test_actual_nrs_coefficient_failure_remains_a_core_failure(self):
        self.fixture(pair=False, offset=.1)
        result = self.snapshot()
        self.assertFalse(result["nrs_correctness_passed"])
        self.assertEqual(result["cases"][0]["agreement"]["status"], "failed")
        self.assertTrue(result["cases"][0]["agreement"]["checks"][0]["failures"])

    def test_failed_and_unresolved_runtime_gates_do_not_become_core_passes(self):
        for ratios, expected in (([1.2]*7, "failed"), ([.9, .95, 1., 1.05, 1.1, 1.15, 1.2], "unresolved")):
            with self.subTest(status=expected):
                self.fixture(ratios=ratios)
                result = self.snapshot()
                self.assertTrue(result["nrs_correctness_passed"])
                self.assertFalse(result["nrs_performance_passed"])
                self.assertFalse(result["nrs_core_validation_passed"])
                self.assertEqual(result["cases"][0]["performance"]["status"], expected)

    def test_valid_status_changed_original_finite_invalidates_timing_only(self):
        self.fixture(baseline_offset=.1)
        result = self.snapshot()
        self.assertTrue(result["nrs_correctness_passed"])
        self.assertFalse(result["nrs_performance_passed"])
        self.assertFalse(result["nrs_core_validation_passed"])
        case = result["cases"][0]
        self.assertTrue(case["performance"]["passed"])
        self.assertFalse(case["timing_agreement"]["passed"])
        self.assertEqual(result["nrs_core_status"], "failed")
        self.assertTrue(case["timing_agreement"]["checks"][0]["failures"])

    def test_external_evidence_can_be_raw_log_without_json_reinterpretation(self):
        self.fixture(pair=False, external=True)
        log = self.root / "external-worker.log"
        log.write_text("Raw retained log: status 410 was rejected.\n")
        bound = report.v2.bound_file(log)
        self.inputs[bound["path"]] = bound["sha256"]
        value = json.loads((self.root / "external.json").read_text())
        value["findings"][0]["evidence"].append(bound)
        self.dataset["external_findings"] = self.write("external.json", value)
        self.dataset["inputs_sha256"] = dict(self.inputs)
        self.dataset_path.write_text(json.dumps(self.dataset))
        result = self.snapshot()
        self.assertEqual(result["external_backend_comparison_status"], "failed")

    def test_changed_raw_bytes_after_authority_validation_rejected(self):
        self.fixture(pair=False)
        filename = next(p for p in (self.root / "correct/data").iterdir())
        filename.write_text("{}")
        with self.assertRaisesRegex(ValueError, "Changed report input"):
            self.snapshot()

    def test_changed_dataset_after_validation_rejected(self):
        self.fixture(pair=False)
        self.dataset_path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "Dataset changed"):
            self.snapshot()

    def test_forged_core_boolean_is_not_trusted(self):
        self.fixture(pair=False)
        self.dataset["nrs_performance_passed"] = True
        self.dataset_path.write_text(json.dumps(self.dataset))
        with self.assertRaisesRegex(ValueError, "core gates"):
            self.snapshot()

    def test_external_failure_cannot_be_relabeled_passed(self):
        self.fixture(external=True, pair=False)
        value = json.loads((self.root / "external.json").read_text())
        value["findings"][0]["status"] = "passed"
        binding = self.write("external.json", value)
        self.dataset["external_findings"] = binding
        self.dataset["inputs_sha256"] = dict(self.inputs)
        self.dataset_path.write_text(json.dumps(self.dataset))
        with self.assertRaisesRegex(ValueError, "relabelled passed"):
            self.snapshot()

    def test_v2_dataset_authority_and_template_globals_remain_untouched(self):
        self.fixture(pair=False)
        old_report = report.legacy.REPORT
        old_roles = copy.deepcopy(report.v2.ROLES)
        with patch.object(report.v2, "validate_dataset", side_effect=AssertionError("v2 dataset used")):
            result = self.snapshot()
        self.assertIs(report.legacy.REPORT, old_report)
        self.assertEqual(report.v2.ROLES, old_roles)
        self.assertFalse(result["presentation"]["full_v2_template_compatible"])
        self.assertFalse(result["legacy_v2_complete_campaign_claim"])

    def test_calibration_relabel_and_failed_status_rejected(self):
        record = self.record("qcd-1", 1, "calibration")
        record["repetition"] = 0
        with self.assertRaisesRegex(ValueError, "calibration"):
            report.calibration_usage(record)
        record = self.record("qcd-1", 1, "calibration")
        record["rows"][0]["status"] = 410
        with self.assertRaises(RuntimeError):
            report.calibration_usage(record)

    def test_seven_pair_timings_preserve_pending_other_backend_cells(self):
        self.fixture()
        case = self.snapshot()["cases"][0]
        self.assertEqual(case["timings"]["candidate"]["8"]["repetitions"], 7)
        self.assertEqual(case["timings"]["candidate"]["8"]["dp"], 700)
        self.assertEqual(case["timings"]["candidate"]["8"]["ratio"], .9)
        self.assertEqual(case["timings"]["candidate"]["7"]["status"], "pending")
        self.assertIsNone(case["timings"]["candidate"]["7"]["median_us"])

    def qualification(self, profiles, precision_result=None, rules=None, **specifications):
        target = {"role": "candidate_corrected", "backend": 8,
                  "generation_source_identity": {"tree_state": {"tracked": {"madgraph/example.py": "source"}}}}
        binding = {"targets": {"qcd-1-candidate-8": target}}
        signature = {"identity_sha256": "runtime", "libraries": {"libninjars.a": "archive"}}
        with ExitStack() as stack:
            stack.enter_context(patch.object(report.legacy, "candidate_runtime_signature", return_value=signature))
            stack.enter_context(patch.object(report.legacy, "runtime_provenance_snapshot", return_value={"status": "passed"}))
            stack.enter_context(patch.object(report.legacy, "regression_snapshot", return_value={"status": "passed"}))
            stack.enter_context(patch.object(report.legacy, "supplement_snapshot", return_value={"status": "passed", "complete_campaign_passed": True}))
            stack.enter_context(patch.object(report.v2, "proof_runtime_signatures", return_value={
                "candidate_corrected/8": {"runtime": {"precisions": ["dp", "qp"]}}}))
            profile_reader = stack.enter_context(patch.object(report.legacy, "profile_snapshot", return_value=profiles))
            precision_reader = stack.enter_context(patch.object(report.precision, "validate", return_value=precision_result or {"status": "passed"}))
            inputs = type("Checked", (), {"checked": {}})()
            value = report.qualification_snapshot({"identity_sha256": "dataset"}, rules or {}, {"entries": []},
                binding, {}, inputs, precision_spec="actual-proof", profile_spec="profile-spec", **specifications)
            return value, profile_reader.call_args, precision_reader.call_args, binding, signature

    def test_nonempty_profile_is_separate_and_uses_only_unique_pair_binding(self):
        profile = {"samples": 10, "evaluations": 100, "role": "candidate_corrected", "backend": 8}
        result, profile_call, precision_call, pair, signature = self.qualification([profile])
        self.assertEqual(result["profile_status"], "passed")
        self.assertEqual(profile_call.args[2], [pair])
        self.assertEqual(precision_call.kwargs["runtime_signature"], signature)
        self.assertEqual(result["qualifications"]["precision_rescue"]["status"], "passed")
        self.assertEqual(result["qualifications"]["installation_export"]["status"], "pending")
        self.assertEqual(result["qualifications"]["focused_controls"]["status"], "pending")

    def test_empty_or_zero_sample_profiles_cannot_satisfy_publication(self):
        result, _, _, _, _ = self.qualification([])
        self.assertEqual(result["profile_status"], "pending")
        with self.assertRaisesRegex(ValueError, "positive actual"):
            self.qualification([{"samples": 0, "evaluations": 100}])
        with self.assertRaisesRegex(ValueError, "positive actual"):
            self.qualification([{"samples": 100, "evaluations": 0}])

    def test_failed_installed_precision_does_not_inherit_representative_pass(self):
        result, _, _, _, _ = self.qualification([], precision_result={"status": "failed"})
        self.assertEqual(result["qualifications"]["precision_rescue"]["status"], "failed")

    def test_explicit_release_requires_separate_source_convergence_authority(self):
        result, _, _, _, _ = self.qualification([], rules={"candidate_cohort": "bound final cohort"})
        precision = result["qualifications"]["precision_rescue"]
        self.assertEqual(precision["installed_precision"]["status"], "passed")
        self.assertTrue(precision["normal_dp_qp_runtime_proven"])
        self.assertTrue(precision["source_precision_required"])
        self.assertEqual(precision["status"], "pending")
        with patch.object(report.release_proofs, "numerical_snapshot", return_value={
                "status": "passed", "historical_bit_identity_test_passed": False,
                "known_provider_regression_test_passed": False}) as reader:
            result, _, _, _, signature = self.qualification([], rules={"candidate_cohort": "bound final cohort"},
                numerical_controls_spec=report.release_proofs.NUMERICAL)
        self.assertEqual(reader.call_args.args[2], signature)
        precision = result["qualifications"]["precision_rescue"]
        self.assertEqual(precision["status"], "passed")
        self.assertFalse(precision["source_precision_controls"]["historical_bit_identity_test_passed"])
        self.assertFalse(precision["source_precision_controls"]["known_provider_regression_test_passed"])

    def test_new_install_software_readers_receive_measured_source_and_runtime(self):
        with patch.object(report.release_proofs, "installation_snapshot", return_value={"status": "passed"}) as install:
            with patch.object(report.release_proofs, "regression_snapshot", return_value={"status": "passed"}) as software:
                result, _, _, binding, signature = self.qualification([], installation_export_spec=report.release_proofs.INSTALLATION,
                    validation_spec=report.release_proofs.REGRESSIONS)
        self.assertEqual(install.call_args.args[1], signature)
        files = binding["targets"]["qcd-1-candidate-8"]["generation_source_identity"]["tree_state"]["tracked"]
        self.assertEqual(install.call_args.args[2], files)
        self.assertEqual(software.call_args.args[1], files)
        self.assertEqual(result["qualifications"]["installation_export"]["status"], "passed")
        self.assertEqual(result["qualifications"]["focused_controls"]["status"], "pending")
        self.assertEqual(result["profile_status"], "pending")

    def pdf_fixture(self, snapshot, name):
        target = self.root / name
        snapshot = copy.deepcopy(snapshot)
        snapshot["generated_utc"] = "Synthetic layout fixture · no physical evaluation"
        with patch.object(report, "build_snapshot", return_value=snapshot):
            result = report.render(self.dataset_path, target,
                                   timestamp="Synthetic layout fixture", compile_pdf=True)
        key = "docs/benchmarkfolder/policy-v3/template/benchmark.typ"
        self.assertEqual(report.TEMPLATE, report.ROOT / key)
        self.assertEqual(result["render_sources_sha256"][key], hashlib.sha256(report.TEMPLATE.read_bytes()).hexdigest())
        self.assertEqual((target / "benchmark.typ").read_bytes(), report.TEMPLATE.read_bytes())
        self.assertEqual((target / "style.typ").read_bytes(), report.STYLE.read_bytes())
        self.assertEqual(result["historical_research_fragment_sha256"], hashlib.sha256((target / "research.typ").read_bytes()).hexdigest())
        self.assertFalse(result["final_publication_passed"])
        text = subprocess.check_output(["pdftotext", "-layout", str(target / "benchmark.pdf"), "-"], text=True)
        retention = os.environ.get("NINJARS_REPORT_V3_SOFTWARE_ARTIFACTS")
        if retention:
            retained = report.v2.path(retention) / name
            self.assertFalse(retained.exists())
            shutil.copytree(target, retained)
        return result, " ".join(text.split()), target

    @unittest.skipUnless(shutil.which("typst") and shutil.which("pdftotext"), "Typst and pdftotext are required")
    def test_v3_pdf_keeps_passing_nrs_and_failed_external_distinct(self):
        self.fixture(external=True)
        result, text, _ = self.pdf_fixture(self.snapshot(), "layout-nrs-external")
        self.assertTrue(result["nrs_core_validation_passed"])
        for value in ("NinjaRS core passed", "External-backend findings failed",
                      "Full qualification pending", "Final publication pending", "17, 52",
                      "3.15%", "9.24%", "Ninja (backend 6)"):
            self.assertIn(value, text)
        self.assertRegex(text, r"9\s+0\s*/\s*0\s+0\.00000\s+passed")
        self.assertIn("700 DP / 0 QP", text)
        self.assertIn("Timing pending", text)

    @unittest.skipUnless(shutil.which("typst") and shutil.which("pdftotext"), "Typst and pdftotext are required")
    def test_v3_pdf_displays_ratio_intervals_and_timing_same_outcome_failure(self):
        self.fixture(baseline_offset=.1)
        snapshot = self.snapshot()
        for case, ratio, ci, status in zip(snapshot["cases"][:3], (.923, 1.234, 1.051),
                                          ([.901, .945], [1.123, 1.345], [1.031, 1.071]),
                                          ("passed", "failed", "unresolved")):
            # Explicit synthetic presentation values, never a scientific snapshot.
            case["performance"].update(runtime_ratio=ratio, ratio_confidence_interval_95=ci, status=status)
        _, text, _ = self.pdf_fixture(snapshot, "layout-intervals-same-outcome")
        table = text.split("unchanged performance gate", 1)[1].split("Threshold", 1)[0]
        self.assertRegex(table, r"0\.923\s+0\.901\s+–\s+0\.945\s+passed\s+failed")
        self.assertRegex(table, r"1\.234\s+1\.123\s+–\s+1\.345\s+failed\s+failed")
        self.assertRegex(table, r"1\.051\s+1\.031\s+–\s+1\.071\s+unresolved\s+failed")
        self.assertIn("Numerical correctness passed", text)
        self.assertIn("NinjaRS core failed", text)
        self.assertIn("Paired performance failed", text)
        self.assertIn("Full qualification failed", text)
        self.assertIn("Final publication failed", text)
        self.fixture(offset=.1)
        _, failed_text, _ = self.pdf_fixture(self.snapshot(), "layout-nrs-correctness-failure")
        self.assertIn("Numerical correctness failed", failed_text)
        self.assertIn("NinjaRS core failed", failed_text)

    @unittest.skipUnless(shutil.which("typst") and shutil.which("pdftotext"), "Typst and pdftotext are required")
    def test_v3_pdf_raw_view_scope_and_deterministic_reproduction(self):
        self.fixture(pair=False)
        self.completion_view()
        snapshot = self.snapshot()
        outputs = []
        for name, epoch in (("layout-reproduction-a", "1"), ("layout-reproduction-b", "999999999")):
            with patch.dict(os.environ, {"SOURCE_DATE_EPOCH": epoch}):
                _, text, target = self.pdf_fixture(snapshot, name)
            outputs.append(target)
            self.assertIn("Original controller failed finalization", text)
            self.assertIn("original status remains failed", text)
            self.assertIn("No new amplitude evaluations, retries or fresh-process parity", text)
            self.assertIn("zero timing repetitions", text)
        self.assertEqual((outputs[0] / "report-data.json").read_bytes(), (outputs[1] / "report-data.json").read_bytes())
        self.assertEqual((outputs[0] / "benchmark.pdf").read_bytes(), (outputs[1] / "benchmark.pdf").read_bytes())

    def test_no_pair_cannot_self_qualify_from_unrelated_proof_paths(self):
        inputs = type("Checked", (), {"checked": {}})()
        with self.assertRaisesRegex(ValueError, "actual pair"):
            report.qualification_snapshot({}, {}, {}, None, {}, inputs, precision_spec="other-proof")
        with self.assertRaisesRegex(ValueError, "actual pair"):
            report.qualification_snapshot({}, {}, {}, None, {}, inputs,
                installation_export_spec=report.release_proofs.INSTALLATION)


if __name__ == "__main__":
    unittest.main()
