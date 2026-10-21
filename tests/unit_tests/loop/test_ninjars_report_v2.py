"""Independent report checks; structural source/worker validation has its own suite."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import ninjars_report_v2 as report


class NinjaRSReportV2Test(unittest.TestCase):
    def setUp(self):
        (report.ROOT / ".ninjars-build/tmp").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=report.ROOT / ".ninjars-build/tmp",
                                                prefix="report-v2-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.policy = report.policy
        self.config = {"cases": [{"id": "qcd-1", "nfinal": 1, "process": "u u~ > z [virt=QCD]",
                                 "model": "loop_sm"}],
                       "points_per_case": 2, "repetitions": 7, "seed": 410083,
                       "max_runtime_ratio": 1.05, "energy_gev": 1000.}

    def write(self, name, value):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
        return p

    def bound(self, name, value):
        return self.policy.bound_file(self.write(name, value))

    def record(self, role, backend, rep, offset=0., ratio=.9):
        source = "candidate" if role == "candidate_corrected" else "baseline"
        code = {1: 211, 6: 216, 7: 217, 8: 218}[backend]
        return {"case": "qcd-1", "source": source, "backend": backend,
                "repetition": rep, "seconds_per_point": ratio if source == "candidate" else 1.,
                "cold_seconds": .5, "loops": 2, "cpu": 207,
                "timed_status_counts": {str(code): 4}, "status_counts": {str(code): 2},
                "executable_sha256": role + str(backend),
                "points_sha256": "points", "parameter_card_sha256": "parameters",
                "rows": [{"point": p, "order": 0, "status": code, "accuracy": 1e-12,
                          "coefficients": [1., 2.+offset, 3., 4.]} for p in (1, 2)]}

    def fixture(self, original_offset=0., reference_offset=0., ratio=.9, pending=False):
        timing = {"report": str(self.root / "timing"), "targets": {}}
        refs = {"report": str(self.root / "refs"), "targets": {}}
        measurements, firsts, times = {}, {}, {}
        for role, backends in self.policy.ROLES.items():
            binding = refs if role == "original_helper_corrected" else timing
            count = 2 if binding is refs else 7
            source = "candidate" if role == "candidate_corrected" else "baseline"
            for backend in backends:
                offset = original_offset if role == "original" else (
                    reference_offset if role == "original_helper_corrected" and backend == 1 else 0.)
                records = [self.record(role, backend, r, offset, ratio) for r in range(count)]
                firsts[f"{role}/{backend}"] = records[0]["rows"]
                times[f"{role}/{backend}"] = [r["seconds_per_point"] for r in records]
                build = self.bound(f"builds/{role}-{backend}.json",
                                   {"origin_compilation": {"seconds": 1., "build_kind": "clean-subprocess"},
                                    "generation_source_identity": {"revision": role},
                                    "parameter_card_sha256": "parameters",
                                    "original_generated_sources_sha256":
                                        report.bench.content_digest({"process.f": role})})
                binding["targets"][f"qcd-1-{source}-{backend}"] = {
                    "role": role, "build": build, "backend": backend}
                for r in records:
                    path = self.write(f"{'refs' if binding is refs else 'timing'}/data/{source}-{backend}-{r['repetition']}.json", r)
                    measurements[str(path.relative_to(report.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        correction = self.bound("correction.json", {
            "identity_sha256": "correction",
            "cases": {"qcd-1": {"original_generation": {
                "seconds": 2., "source_identity": {"revision": "original"},
                "parameter_card_sha256": "parameters"},
                "generation_records": [None, None, self.bound("generation.json", {
                    "seconds": 3., "source_identity": {"revision": "candidate_corrected"},
                    "parameter_card_sha256": "parameters"})]}}})
        policy = self.bound("policy.json", {"identity_sha256": "policy", "correction": correction,
                                          "configuration": self.bound("config.json", self.config)})
        corrected = {k: v for k, v in firsts.items() if not k.startswith("original/")}
        pairs = [(f"original/{b}", f"original_helper_corrected/{b}") for b in (1, 6, 7)]
        pairs += [(f"original/{b}", f"candidate_corrected/{b}") for b in (1, 6, 7)]
        pairs += [("original/6", "candidate_corrected/8")]
        agreement = self.policy.compare_rows(corrected)
        performance = report.bench.runtime_statistics(times["original/6"], times["candidate_corrected/8"], 1.05, 410083)
        dataset = {"identity_sha256": "dataset", "policy": policy,
                   "representative_proof": self.bound("proof.json", {"identity_sha256": "proof"}),
                   "timing_bindings": self.bound("timing-bindings.json", timing),
                   "reference_bindings": self.bound("reference-bindings.json", refs),
                   "measurements_sha256": measurements, "completed_cases": ["qcd-1"],
                   "cases": {"qcd-1": {"corrected_agreement": agreement,
                                      "original_diagnostics": self.policy.compare_rows(firsts, pairs),
                                      "performance": performance}},
                   "all_cases_passed": agreement["passed"] and performance["passed"]}
        if pending:
            self.config["cases"].append({"id": "ew-1", "model": "loop_qcd_qed_sm_Gmu",
                                         "nfinal": 1, "process": "u d~ > w+ [virt=QED]"})
            self.write("config.json", self.config)
            rules = json.loads((self.root / "policy.json").read_text())
            rules["configuration"] = self.policy.bound_file(self.root / "config.json")
            dataset["policy"] = self.bound("policy.json", rules)
            dataset["all_cases_passed"] = False
        self.dataset = dataset
        self.dataset_path = self.write("dataset.json", dataset)
        return dataset

    def snapshot(self, generation_inputs=None):
        with patch.object(self.policy, "validate_dataset", return_value=self.dataset):
            return report.build_snapshot(self.dataset_path, timestamp="test",
                                         generation_inputs=generation_inputs)

    def test_generation_from_older_corrected_export_stays_pending(self):
        self.fixture()
        # The model-helper proof remains valid after a later generator change.
        rules = json.loads((self.root / "policy.json").read_text())
        correction = json.loads((self.root / "correction.json").read_text())
        record = json.loads((self.root / "generation.json").read_text())
        record["source_identity"] = {"revision": "older-candidate"}
        correction["cases"]["qcd-1"]["generation_records"][2] = self.bound("generation.json", record)
        rules["correction"] = self.bound("correction.json", correction)
        self.dataset["policy"] = self.bound("policy.json", rules)
        result = self.snapshot()["cases"][0]["preparation"]
        self.assertIsNone(result["candidate"]["generation"])
        self.assertEqual(result["baseline"]["generation"]["seconds"], 2.)
        self.assertEqual(result["candidate"]["compilation"]["seconds"], 1.)

    def test_final_generation_metadata_binds_exact_measured_sources(self):
        self.fixture()
        root = self.root / "final-inputs"
        record = {"seconds": 19., "source_identity": {"revision": "candidate_corrected"},
                  "parameter_card_sha256": "parameters"}
        manifest = {"files": {"process.f": "candidate_corrected"}}
        manifest["files_sha256"] = report.bench.content_digest(manifest["files"])
        self.write("final-inputs/candidate/qcd-1/generation.json", record)
        self.write("final-inputs/candidate/qcd-1/compiled-sources.json", manifest)
        result = self.snapshot(root)
        self.assertEqual(result["cases"][0]["preparation"]["candidate"]["generation"]["seconds"], 19.)
        self.assertEqual(result["generation_inputs"], self.policy.relative(root))
        retained = str((root / "candidate/qcd-1/generation.json").relative_to(report.ROOT))
        self.assertIn(retained, result["inputs_sha256"])
        for field, value, message in (
                ("source_identity", {"revision": "wrong"}, "identity"),
                ("parameter_card_sha256", "wrong", "parameter card"),
                ("seconds", float("nan"), "duration")):
            with self.subTest(field=field):
                self.write("final-inputs/candidate/qcd-1/generation.json", {**record, field: value})
                with self.assertRaisesRegex(ValueError, message):
                    self.snapshot(root)
        self.write("final-inputs/candidate/qcd-1/generation.json", record)
        manifest["files"]["process.f"] = "different-case"
        manifest["files_sha256"] = report.bench.content_digest(manifest["files"])
        self.write("final-inputs/candidate/qcd-1/compiled-sources.json", manifest)
        with self.assertRaisesRegex(ValueError, "source manifest"):
            self.snapshot(root)

    def test_ordinary_collector_generation_inputs_are_found_automatically(self):
        self.fixture()
        self.write("timing/inputs/candidate/qcd-1/generation.json", {
            "seconds": 23., "source_identity": {"revision": "candidate_corrected"},
            "parameter_card_sha256": "parameters"})
        files = {"process.f": "candidate_corrected"}
        self.write("timing/inputs/candidate/qcd-1/compiled-sources.json", {
            "files": files, "files_sha256": report.bench.content_digest(files)})
        result = self.snapshot()["cases"][0]["preparation"]
        self.assertEqual(result["candidate"]["generation"]["seconds"], 23.)

    def test_seven_corrected_targets_and_original_disagreements_remain_distinct(self):
        self.fixture(original_offset=.01)
        result = self.snapshot()
        case = result["cases"][0]
        self.assertEqual(case["agreement"]["pairs"], 21)
        self.assertEqual(case["agreement"]["comparisons"], 168)
        self.assertEqual(case["agreement"]["status"], "passed")
        self.assertEqual(case["original_diagnostics"]["status"], "failed")
        self.assertGreater(result["summary"]["original_mismatch_records"], 0)
        self.assertEqual(case["performance"]["runtime_ratio"], .9)
        self.assertEqual(case["timings"]["baseline"]["6"]["median_us"], 1e6)
        self.assertEqual(case["corrected_reference_usage"]["6"]["repetitions"], 2)

    def test_reference_disagreement_fails_even_if_four_candidates_agree(self):
        self.fixture(reference_offset=.01)
        result = self.snapshot()
        self.assertEqual(result["summary"]["correctness_failed"], 1)
        self.assertFalse(result["all_cases_passed"])

    def test_forged_summary_and_performance_are_rejected(self):
        self.fixture(original_offset=.01)
        self.dataset["cases"]["qcd-1"]["original_diagnostics"]["passed"] = True
        with self.assertRaisesRegex(ValueError, "forged"):
            self.snapshot()
        self.fixture()
        self.dataset["cases"]["qcd-1"]["performance"]["runtime_ratio"] = .1
        with self.assertRaisesRegex(ValueError, "performance"):
            self.snapshot()

    def test_unmeasured_configured_cases_stay_pending(self):
        self.fixture(pending=True)
        result = self.snapshot()
        self.assertEqual(result["summary"]["cases"], 2)
        self.assertEqual(result["summary"]["correctness_passed"], 1)
        self.assertEqual(result["cases"][1]["agreement"]["status"], "pending")
        self.assertIsNone(result["cases"][1]["timings"]["candidate"]["8"]["median_us"])
        self.assertFalse(result["all_cases_passed"])
        self.assertEqual(result["cases"][1]["stability"], {"status": "pending", "targets": {}})

    def test_core_pass_does_not_complete_missing_supplement(self):
        self.fixture()
        result = self.snapshot()
        self.assertTrue(result["all_cases_passed"])
        self.assertFalse(result["complete_campaign_passed"])
        self.assertEqual(result["campaign_status"], "pending")
        self.assertEqual(result["supplement"]["status"], "pending")
        self.assertEqual(result["regressions"]["status"], "pending")
        self.assertIsNone(result["supplement_specification"])
        for key in ("ninjars_revision", "oneloopmaster_revision", "symbolica_revision",
                    "ninjars_abi_version", "accepted_core_series"):
            self.assertNotIn(key, result["provenance"])
        self.assertIsNone(result["runtime_provenance"]["dependencies"]["ninjars"])
        self.assertFalse(result["complete_campaign_passed"])

    def test_core_stability_separates_estimates_from_observed_discrepancy(self):
        self.fixture(original_offset=1e20)
        stability = self.snapshot()["cases"][0]["stability"]
        self.assertFalse(stability["estimate_is_observed_error"])
        self.assertEqual(len(stability["targets"]), 7)
        candidate = stability["targets"]["candidate_corrected/8"]
        self.assertEqual(candidate["estimated_accuracy"]["all_orders"]["maximum"], 1e-12)
        self.assertEqual(candidate["observed_discrepancy"]["max_absolute_error"], 0.)
        self.assertNotIn("original/6", candidate["observed_discrepancy"]["reference_targets"])
        self.assertEqual(candidate["status_counts"], {"218": 2})

    def test_complete_campaign_requires_additional_stress_controls(self):
        self.fixture()
        with patch.object(report, "supplement_snapshot", return_value={
                "status": "passed", "complete_campaign_passed": True}), \
                patch.object(report, "runtime_provenance_snapshot", return_value={"status": "passed"}), \
                patch.object(report, "regression_snapshot", return_value={"status": "passed"}):
            result = self.snapshot()
            self.assertEqual(result["stress"]["status"], "pending")
            self.assertFalse(result["complete_campaign_passed"])
            for status in ("failed", "unresolved", "passed"):
                stress = {"status": status, "complete_campaign_passed": status == "passed", "cases": []}
                with self.subTest(status=status), patch.object(report, "stress_snapshot", return_value=stress):
                    result = self.snapshot()
                    self.assertEqual(result["complete_campaign_passed"], status == "passed")
                    self.assertEqual(result["campaign_status"], status)

    def test_rejected_dataset_cannot_render_or_create_output(self):
        target = self.root / "report"
        with patch.object(self.policy, "validate_dataset", side_effect=RuntimeError("invalid proof")):
            with self.assertRaisesRegex(RuntimeError, "invalid proof"):
                report.render(self.root / "absent.json", target)
        self.assertFalse(target.exists())

    def test_report_keeps_failed_gate_and_never_overwrites(self):
        self.fixture(ratio=1.2)
        target = self.root / "report"
        with patch.object(self.policy, "validate_dataset", return_value=self.dataset):
            result = report.render(self.dataset_path, target, timestamp="test")
        self.assertEqual(result["summary"]["performance_failed"], 1)
        self.assertEqual(result["campaign_status"], "failed")
        self.assertTrue((target / "benchmark.typ").exists())
        with self.assertRaisesRegex(ValueError, "exists"):
            report.render(self.dataset_path, target)

    def test_stress_render_hashes_actual_reader_and_transitive_supplement_source(self):
        self.fixture()
        snapshot = self.snapshot()
        specification = self.write("synthetic-stress.json", {})
        target = self.root / "stress-source-hashes"
        with patch.object(report, "build_snapshot", return_value=snapshot):
            result = report.render(self.dataset_path, target, stress_spec=specification)
        for name in ("tools/ninjars_stress_v2.py", "tools/ninjars_supplement_v2.py"):
            self.assertEqual(result["render_sources_sha256"][name],
                             report.workers.digest(report.ROOT / name))

    @unittest.skipUnless(shutil.which("typst"), "Typst is required for PDF reproducibility")
    def test_pdf_is_identical_across_ambient_creation_epochs(self):
        self.fixture()
        reports = []
        for index, epoch in enumerate(("0", "1893456000")):
            target = self.root / f"reproducible-pdf-{index}"
            with patch.object(self.policy, "validate_dataset", return_value=self.dataset), \
                    patch.dict(os.environ, {"SOURCE_DATE_EPOCH": epoch}):
                report.render(self.dataset_path, target,
                              timestamp="2026-10-20 · 00:00 UTC", compile_pdf=True)
            reports.append(target)
        self.assertEqual((reports[0] / "report-data.json").read_bytes(),
                         (reports[1] / "report-data.json").read_bytes())
        self.assertEqual((reports[0] / "benchmark.pdf").read_bytes(),
                         (reports[1] / "benchmark.pdf").read_bytes())

    @unittest.skipUnless(shutil.which("typst"), "Typst is required for supplement layout")
    def test_supplement_and_threshold_pages_compile_with_explicit_synthetic_fixture(self):
        self.fixture()
        snapshot = self.snapshot()
        prototype = copy.deepcopy(snapshot["cases"][0])
        snapshot["cases"] = [dict(copy.deepcopy(prototype), id=f"{kind}-{number}", nfinal=number)
                             for kind in ("qcd", "ew") for number in range(1, 5)]
        snapshot["summary"].update(cases=8, measured_candidate_cells=32, correctness_passed=8,
                                  performance_passed=8)
        loop_cases = [dict(copy.deepcopy(prototype), id=cid, process=process, points=2, nfinal=nfinal)
                      for cid, process, nfinal in (("supplement-gg-h", "g g > h [sqrvirt=QCD]", 1),
                                                   ("supplement-gg-hh", "g g > h h [sqrvirt=QCD]", 2))]
        snapshot["supplement"] = {
            "status": "pending", "complete_campaign_passed": False,
            "cases": loop_cases,
            "threshold_grid": {
                "agreement": {"status": "passed"},
                "point_summaries": [{"point": point, "delta": delta, "cos_theta": cosine,
                                     "max_tolerance_fraction": .2,
                                     "estimated_accuracy": {"candidate_corrected/8": 5e-324 if point == 1 else 1e-12}}
                                    for point, (delta, cosine) in enumerate(
                                        ((d, c) for d in ("-0.01", "-0.0001", "-0.000001",
                                                         "0.000001", "0.0001", "0.01")
                                         for c in ("0", "0.8")), 1)]},
            "precision_controls": {"status": "pending", "working_bits": 256,
                                   "convergence_bits": 384,
                                   "checks": [{"name": "Explicit synthetic layout fixture", "status": "pending"}]}}
        snapshot["regressions"] = {
            "status": "passed", "tests_run": 2, "tests_passed": 1, "total_skipped": 1,
            "scope": "Explicit synthetic layout fixture; no physical process was evaluated.",
            "suites": [{"name": "Explicit synthetic suite", "tests_run": 2, "failures": 0,
                        "errors": 0, "skipped": 1,
                        "skip_details": [["synthetic.case", "Explicit synthetic skip"]]}]}
        stress = {"points": 20,
                  "ordinary": {"agreement": prototype["agreement"], "stability": prototype["stability"],
                               "diagnostic_timings": prototype["timings"]["candidate"]},
                  "rotation": {"points": 4, "agreement": prototype["agreement"],
                               "invariance": prototype["agreement"]},
                  "scale": {"points": 8, "agreement": prototype["agreement"],
                            "restoration": prototype["agreement"]},
                  "optional_reference": {"status": "unavailable", "agreement": None,
                                         "failures": ["Explicit synthetic unavailable reference"]}}
        snapshot["stress"] = {"status": "pending", "complete_campaign_passed": False,
                              "cases": [dict(copy.deepcopy(stress), id=cid) for cid in
                                        ("stress-qcd-4-gg-ttgg", "stress-ew-4-uu-4l")]}
        # A later failed optional helper collection must remain visible even if
        # an independently complete helper cohort had already agreed.
        snapshot["stress"]["cases"][1]["optional_reference"] = {
            "status": "failed", "agreement": prototype["agreement"],
            "failures": ["Explicit synthetic retained helper failure"]}
        target = self.root / "supplement-layout-fixture"
        with patch.object(report, "build_snapshot", return_value=snapshot):
            result = report.render(self.dataset_path, target, compile_pdf=True)
        self.assertFalse(result["complete_campaign_passed"])
        self.assertTrue((target / "benchmark.pdf").read_bytes().startswith(b"%PDF"))

    def _presentation_snapshot(self):
        """Explicit synthetic layout data; never a scientific campaign result."""
        self.fixture()
        snapshot = self.snapshot()
        prototype = snapshot["cases"][0]
        snapshot["supplement"]["cases"] = [
            dict(copy.deepcopy(prototype), id=cid, process=process, points=100,
                 nfinal=nfinal)
            for cid, process, nfinal in (
                ("supplement-gg-h", "g g > h [sqrvirt=QCD]", 1),
                ("supplement-gg-hh", "g g > h h [sqrvirt=QCD]", 2))]
        snapshot["generated_utc"] = "Explicit synthetic presentation test"
        self.assertFalse(snapshot["complete_campaign_passed"])
        return snapshot

    def _presentation_pdf_text(self, snapshot, name):
        target = self.root / name
        with patch.object(report, "build_snapshot", return_value=snapshot):
            result = report.render(self.dataset_path, target, compile_pdf=True)
        self.assertFalse(result["complete_campaign_passed"])
        template = report.REPORT / "policy-v2/template/benchmark.typ"
        expected = hashlib.sha256(template.read_bytes()).hexdigest()
        self.assertEqual(result["render_sources_sha256"][str(template.relative_to(report.ROOT))],
                         expected)
        self.assertEqual((target / "benchmark.typ").read_bytes(), template.read_bytes())
        text = subprocess.check_output(["pdftotext", "-layout", str(target / "benchmark.pdf"), "-"],
                                       text=True)
        return " ".join(text.split())

    @unittest.skipUnless(shutil.which("typst") and shutil.which("pdftotext"),
                         "Typst and pdftotext are required for presentation assertions")
    def test_loop_induced_pdf_displays_all_backend_dp_qp_counts(self):
        snapshot = self._presentation_snapshot()
        expected = []
        for i, case in enumerate(snapshot["supplement"]["cases"]):
            for j, backend in enumerate(("1", "6", "7", "8")):
                dp, qp = 2101 + 100*i + j, 3101 + 100*i + j
                case["timings"]["candidate"][backend].update(dp=dp, qp=qp,
                                                            timed_calls=dp+qp)
                expected.extend((f"{dp} DP", f"{qp} QP"))
        text = self._presentation_pdf_text(snapshot, "synthetic-loop-usage")
        usage = text.split("Timed double- and higher-precision evaluations", 1)[1].split(
            "Backward-compatible status interpretation", 1)[0]
        for pair in expected:
            self.assertIn(pair, usage)

    @unittest.skipUnless(shutil.which("typst") and shutil.which("pdftotext"),
                         "Typst and pdftotext are required for presentation assertions")
    def test_loop_induced_pdf_displays_ratio_interval_and_all_decisions(self):
        snapshot = self._presentation_snapshot()
        h, hh = snapshot["supplement"]["cases"]
        for case, ratio, interval, status in (
                (h, .923, [.901, .945], "passed"),
                (hh, 1.234, [1.123, 1.345], "failed")):
            case["performance"].update(runtime_ratio=ratio,
                                      ratio_confidence_interval_95=interval, status=status)
        text = self._presentation_pdf_text(snapshot, "synthetic-loop-decisions")
        table = text.split("Performance acceptance", 1)[1].split("Acceptance threshold", 1)[0]
        self.assertRegex(table, r"0\.923\s+0\.901\s+–\s+0\.945\s+passed")
        self.assertRegex(table, r"1\.234\s+1\.123\s+–\s+1\.345\s+failed")
        hh["performance"].update(runtime_ratio=1.051,
                                 ratio_confidence_interval_95=[1.031, 1.071],
                                 status="unresolved")
        text = self._presentation_pdf_text(snapshot, "synthetic-loop-unresolved")
        table = text.split("Performance acceptance", 1)[1].split("Acceptance threshold", 1)[0]
        self.assertRegex(table, r"1\.051\s+1\.031\s+–\s+1\.071\s+unresolved")

    @unittest.skipUnless(shutil.which("typst") and shutil.which("pdftotext"),
                         "Typst and pdftotext are required for presentation assertions")
    def test_pdf_optimization_attribution_matches_retained_source_and_scopes(self):
        research = report.ROOT / "docs/benchmarkfolder/experiments"
        bindings = {
            "ninjars-mp-tensor-allocation-process-v2/timings/attempt-1/analysis.json":
                "79e661e62a4ce253181bdff22fa1dd530269ccbe570b61e2e2b9c9712e3f1a4d",
            "ninjars-mp-tensor-allocation-process-v2/frozen-bindings.json":
                "64766a546d751705993e4b074f830d9ff32260c89e4ecfd6f6e4aa9114e2871e",
            "ninjars-mp-tensor-allocation-transport-v1/production-tensor.rs":
                "240bbc1f9c8937393227c16a6ce4887ee68524a3f021a8a93656151159445d70",
            "ninjars-native-float256-v1/selection-analysis.json":
                "0581c1fe7ac1371197610c894a360a8ccd82798a3970a167b9d6fc3002d5044c"}
        for relative, digest in bindings.items():
            self.assertEqual(hashlib.sha256((research / relative).read_bytes()).hexdigest(), digest)
        allocation = json.loads((research / next(iter(bindings))).read_text())
        qcd = allocation["cases"]["qcd-3"]
        self.assertEqual(qcd["pairs"], 21)
        self.assertAlmostEqual(qcd["ratio"], .968518392915057)
        self.assertEqual(qcd["confidence_interval_95"], [.9568779827192503, .993639704088713])
        self.assertTrue(allocation["promotion_gates_passed"])
        self.assertFalse(allocation["acceptance_eligible"])
        native = json.loads((research / "ninjars-native-float256-v1/selection-analysis.json").read_text())
        self.assertEqual(native["fixed_triplets"], 7)
        self.assertEqual(native["provisional_request_screen_choice"], "Interpreted")
        self.assertFalse(native["production_change_authorized_by_this_analysis"])
        for arm in native["original_comparisons_vs_interpreted"].values():
            self.assertFalse(arm["demonstrated_request_scope_gain"])
            self.assertLess(arm["paired_bootstrap_95_interval"][0], 1.)
            self.assertGreater(arm["paired_bootstrap_95_interval"][1], 1.)
        text = self._presentation_pdf_text(self._presentation_snapshot(),
                                           "synthetic-research-attribution")
        section = text.split("Measured optimizations and retained negative results", 1)[1].split(
            "Generator, Fortran and export regressions", 1)[0]
        for phrase in ("3.19%", "Ninja (backend 6)", "3.15% full MadLoop QCD3",
                       "0.968518", "0.956878", "0.993640", "9.24% callback-only",
                       "DP implementation is not promoted", "request screen is rejected",
                       "1.021698", "1.007070", "Interpreted execution is retained",
                       "not a full MadLoop timing", "are not added together",
                       "832fc0d11988", "366ae91da891", "historical timing identities",
                       "final installation 800a941faf3c",
                       "b44d20e725f1ec68589df20284e4ce9116e9e2ab",
                       "f4bffc61ed67e5d083b80ff9901768d4d5779989",
                       "public-manager IPC are excluded", "neither improvement nor slowdown"):
            self.assertIn(phrase, section)
        compact = "".join(section.split())
        for relative, digest in bindings.items():
            self.assertIn(relative, compact)
            self.assertIn(digest, compact)

    def test_raw_coefficient_stream_and_status_corruption_rejected(self):
        rows = {"a": self.record("original", 6, 0)["rows"],
                "b": self.record("candidate_corrected", 8, 0)["rows"]}
        invalid = copy.deepcopy(rows)
        invalid["b"].pop()
        with self.assertRaises(ValueError):
            report.compare(invalid)
        invalid = copy.deepcopy(rows)
        invalid["b"][0]["coefficients"].pop()
        with self.assertRaises(ValueError):
            report.compare(invalid)
        invalid = copy.deepcopy(rows)
        invalid["b"][0]["coefficients"][0] = float("nan")
        with self.assertRaises(ValueError):
            report.compare(invalid)
        record = self.record("candidate_corrected", 8, 0)
        record["timed_status_counts"] = {"410": 4}
        with self.assertRaises(RuntimeError):
            report.summary([dict(record, repetition=i) for i in range(7)], 2, 7)

    def test_point_histogram_is_recomputed_and_not_labelled_an_accuracy_distribution(self):
        record = self.record("candidate_corrected", 8, 0)
        runs = [dict(record, repetition=i) for i in range(7)]
        summary = report.summary(runs, 2, 7)
        self.assertEqual(summary["point_status_counts"], {"218": 14})
        self.assertNotIn("accuracy_counts", summary)
        runs[0] = dict(runs[0], status_counts={"322": 2})
        with self.assertRaisesRegex(ValueError, "histogram differs"):
            report.summary(runs, 2, 7)

    def test_wrong_original_magnitude_does_not_relax_corrected_comparison(self):
        self.fixture(original_offset=1e20, reference_offset=.01)
        self.assertEqual(self.snapshot()["cases"][0]["agreement"]["status"], "failed")

    def test_changed_raw_file_is_rejected_after_dataset_verification(self):
        self.fixture()
        filename = next(iter(self.dataset["measurements_sha256"]))
        self.policy.path(filename).write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed after"):
            self.snapshot()


class NinjaRSProfileReportTest(unittest.TestCase):
    """Exercise retained CPU samples without launching numerical workers."""

    def setUp(self):
        (report.ROOT / ".ninjars-build/tmp").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=report.ROOT / ".ninjars-build/tmp",
                                                prefix="profile-report-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        evidence = report.REPORT / "policy-v2/profiles"
        name = "profile-controller-original-qcd1-retry2-20261020"
        self.directory = self.root / "profile"
        shutil.copytree(evidence / name, self.directory)
        self.profile_path = self.directory / "profile.json"
        self.profile = report.policy.read(self.profile_path)
        self.binding = report.policy.read(report.policy.check_file(self.profile["binding"]))
        self.correction = report.policy.validate_correction(
            report.REPORT / "policy-v2/model-correction.json")
        self.dataset = {"measurements_sha256": {
            self.profile["measurement"]["path"]: self.profile["measurement"]["sha256"]}}
        self.raw = evidence / "controller-evidence-20261020/raw-perf" / (name + ".perf")
        self.inputs = {}

    def evaluate(self):
        self.profile_path.write_text(json.dumps(self.profile))
        spec = self.root / "spec.json"
        spec.write_text(json.dumps(report.policy.seal({
            "schema": "profile-report-evidence-v2",
            "profiles": [{"profile": report.policy.bound_file(self.profile_path),
                          "raw_perf": report.policy.bound_file(self.raw)}]})))
        return report.profile_snapshot(spec, self.dataset, [self.binding],
                                       self.correction, self.inputs)

    def test_recompute_retained_samples_and_all_raw_blocks(self):
        result = self.evaluate()[0]
        self.assertEqual(result["samples"], 405)
        self.assertEqual(result["evaluations"], 90000)
        self.assertEqual(result["points"], 100)
        self.assertEqual(result["status_counts"], {"216": 90000})
        self.assertEqual(result["categories"]["fortran_dp_numerator"]["samples"], 194)
        self.assertIn(report.policy.relative(self.raw), self.inputs)

    def test_other_dataset_and_changed_worker_are_rejected(self):
        self.dataset["measurements_sha256"] = {}
        with self.assertRaisesRegex(ValueError, "this dataset"):
            self.evaluate()
        self.dataset["measurements_sha256"][self.profile["measurement"]["path"]] = self.profile["measurement"]["sha256"]
        self.binding["targets"][self.profile["target"]]["executable_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "source/runtime"):
            self.evaluate()

    def test_changed_raw_values_are_not_hidden_by_pass_flags(self):
        values = self.directory / "block-1/values.dat"
        lines = values.read_text().splitlines()
        fields = lines[0].split()
        fields[3] = str(float(fields[3]) + 1.)
        lines[0] = " ".join(fields)
        values.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(ValueError, "physical rows"):
            self.evaluate()

    def test_forged_sample_shares_and_missing_raw_recording_are_rejected(self):
        self.profile["partition"]["categories"]["fortran_dp_numerator"]["percent"] = 0.
        with self.assertRaisesRegex(ValueError, "classification"):
            self.evaluate()
        self.profile = report.policy.read(self.profile_path)
        self.raw = self.root / "wrong.perf"
        self.raw.write_bytes(b"unrelated recording")
        with self.assertRaisesRegex(ValueError, "Changed profile evidence"):
            self.evaluate()

    def test_incomplete_or_acceptance_labelled_profiles_are_rejected(self):
        for key, value in (("complete", False), ("acceptance_eligible", True)):
            saved = self.profile[key]
            self.profile[key] = value
            with self.assertRaisesRegex(ValueError, "diagnostic profile"):
                self.evaluate()
            self.profile[key] = saved


class NinjaRSStabilityReportTest(unittest.TestCase):
    def groups(self):
        rows = [{"point": point, "order": order, "status": 218,
                 "accuracy": 0. if order == 0 else 1e-8,
                 "coefficients": [0., 2., 0., 0.]}
                for point in (1, 2) for order in (0, 1)]
        reference = copy.deepcopy(rows)
        for row in reference:
            row["status"] = 216
        return {"candidate_corrected/8": rows, "original_helper_corrected/6": reference}

    def test_split_orders_do_not_multiply_point_or_rescue_counts(self):
        groups = self.groups()
        for row in groups["candidate_corrected/8"]:
            if row["point"] == 2:
                row["status"] = 322
        result = report.stability_snapshot(groups)["targets"]["candidate_corrected/8"]
        self.assertEqual(result["points"], 2)
        self.assertEqual((result["dp"], result["qp"]), (1, 1))
        self.assertEqual(result["status_counts"], {"218": 1, "322": 1})
        self.assertEqual(result["estimated_accuracy"]["all_orders"]["count"], 4)
        self.assertEqual(result["estimated_accuracy"]["total_order"]["zero_count"], 2)
        self.assertEqual(result["estimated_accuracy"]["total_order"]["maximum"], 0.)

    def test_small_estimate_cannot_hide_actual_mismatch(self):
        groups = self.groups()
        groups["candidate_corrected/8"][0]["accuracy"] = 0.
        groups["candidate_corrected/8"][0]["coefficients"][1] += .01
        result = report.stability_snapshot(groups)["targets"]["candidate_corrected/8"]
        self.assertEqual(result["estimated_accuracy"]["total_order"]["maximum"], 0.)
        observed = result["observed_discrepancy"]
        self.assertGreater(observed["failures"], 0)
        self.assertGreater(observed["max_tolerance_fraction"], 1.)
        self.assertEqual(observed["worst"]["point"], 1)
        self.assertEqual(observed["worst"]["order"], 0)
        self.assertEqual(observed["worst"]["coefficient"], 1)

    def test_bad_estimates_and_inconsistent_point_status_rejected(self):
        for estimate in (-1., float("nan"), float("inf"), True):
            with self.subTest(estimate=estimate):
                groups = self.groups()
                groups["candidate_corrected/8"][0]["accuracy"] = estimate
                with self.assertRaisesRegex(ValueError, "estimate"):
                    report.stability_snapshot(groups)
        groups = self.groups()
        groups["candidate_corrected/8"][1]["status"] = 322
        with self.assertRaisesRegex(ValueError, "Point status"):
            report.stability_snapshot(groups)

    def test_uncorrected_original_cannot_relax_stability_scale(self):
        groups = self.groups()
        groups["original/6"] = groups.pop("original_helper_corrected/6")
        with self.assertRaisesRegex(ValueError, "Uncorrected original"):
            report.stability_snapshot(groups)

    def test_nearest_rank_percentile_and_zero_estimate_are_explicit(self):
        result = report.estimate_distribution([0.] + list(range(1, 20)))
        self.assertEqual(result["p95_nearest_rank"], 18)
        self.assertEqual(result["zero_count"], 1)
        self.assertEqual(result["median"], 9.5)
        self.assertEqual(report.estimate_distribution([0., 0.])["maximum"], 0.)


class NinjaRSReportSupplementBoundaryTest(unittest.TestCase):
    def test_precision_reader_receives_actual_full_runtime_descriptor(self):
        expected = {"core_dataset_identity_sha256": "core",
                    "literal_baseline_revision": report.BASELINE_REVISION,
                    "candidate_generator_identity_sha256": "generator",
                    "candidate_runtime_identity_sha256": "runtime"}
        signature = {"identity_sha256": "runtime", "libraries": {"libninjars.a": "archive"},
                     "metadata": {"ninjars-config.json": "config"}}
        seen = []

        def snapshot(path, cohort, inputs, *, runtime_signature):
            seen.append(runtime_signature)
            return dict(cohort, schema="madloop-final-supplement-v1", status="pending",
                        complete_campaign_passed=False)

        module = SimpleNamespace(report_snapshot=snapshot)
        (report.ROOT / ".ninjars-build/tmp").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=report.ROOT / ".ninjars-build/tmp") as directory:
            manifest = Path(directory) / "supplement.json"
            manifest.write_text("{}")
            inputs = {}
            with patch.dict(sys.modules, {"ninjars_supplement_v2": module}), \
                    patch.object(report, "supplement_expectations", return_value=expected), \
                    patch.object(report, "candidate_runtime_signature", return_value=signature):
                result = report.supplement_snapshot(manifest, {}, [], inputs)
            self.assertEqual(seen, [signature])
            self.assertFalse(result["complete_campaign_passed"])
            self.assertEqual(inputs[report.policy.relative(manifest)], report.workers.digest(manifest))

    def test_cohort_identity_includes_generator_runtime_and_literal_baseline(self):
        item = {"role": "candidate_corrected", "backend": 8,
                "generation_source_identity": {"tree_state_sha256": "generator"},
                "signature": {"identity_sha256": "runtime"}}
        binding = {"targets": {"case-candidate-8": item}}
        with patch.object(report.policy, "reducer_signature", side_effect=lambda value: value["signature"]):
            expected = report.supplement_expectations({"identity_sha256": "core"}, [binding])
            self.assertEqual(expected, {
                "core_dataset_identity_sha256": "core",
                "literal_baseline_revision": "27147b3300f556a75661a886e92bdbe00558ee25",
                "candidate_generator_identity_sha256": "generator",
                "candidate_runtime_identity_sha256": "runtime"})
            for field in ("generator", "runtime"):
                changed = copy.deepcopy(item)
                if field == "generator":
                    changed["generation_source_identity"]["tree_state_sha256"] = "changed"
                else:
                    changed["signature"]["identity_sha256"] = "changed"
                binding["targets"]["other-candidate-8"] = changed
                with self.assertRaisesRegex(ValueError, "one bound"):
                    report.supplement_expectations({"identity_sha256": "core"}, [binding])
                binding["targets"].pop("other-candidate-8")

    def test_supplement_cannot_self_certify_a_different_or_contradictory_cohort(self):
        expected = {"core_dataset_identity_sha256": "core",
                    "literal_baseline_revision": report.BASELINE_REVISION,
                    "candidate_generator_identity_sha256": "generator",
                    "candidate_runtime_identity_sha256": "runtime"}
        valid = dict(expected, schema="madloop-final-supplement-v1", status="passed",
                     complete_campaign_passed=True)
        module = SimpleNamespace(report_snapshot=lambda *args, **kwargs: valid)
        with patch.dict(sys.modules, {"ninjars_supplement_v2": module}), \
                patch.object(report, "supplement_expectations", return_value=expected), \
                patch.object(report, "candidate_runtime_signature", return_value={"identity_sha256": "runtime"}):
            for key in expected:
                changed = dict(valid, **{key: "other"})
                module.report_snapshot = lambda *args, **kwargs: changed
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "differently bound"):
                    report.supplement_snapshot("unused.json", {}, [], {})
            module.report_snapshot = lambda *args, **kwargs: dict(valid, status="failed")
            with self.assertRaisesRegex(ValueError, "Contradictory"):
                report.supplement_snapshot("unused.json", {}, [], {})


class NinjaRSStressReportBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.expected = {"core_dataset_identity_sha256": "core",
                         "literal_baseline_revision": report.BASELINE_REVISION,
                         "candidate_generator_identity_sha256": "generator",
                         "candidate_runtime_identity_sha256": "runtime"}
        self.signature = {"identity_sha256": "runtime", "libraries": {"libninjars.a": "archive"},
                          "metadata": {"ninjars-config.json": "config"}}

    def test_stress_reader_gets_full_runtime_without_a_new_timing_gate(self):
        seen = []

        def read(path, expected, inputs, *, runtime_signature):
            seen.append(runtime_signature)
            return dict(expected, schema="madloop-final-stress-v1", status="passed",
                        complete_campaign_passed=True, cases=[{
                            "ordinary": {"diagnostic_timings": {"8": {"median_us": 1e12}}},
                            "optional_reference": {"status": "unavailable", "failures": ["fixture"]}}])

        (report.ROOT / ".ninjars-build/tmp").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=report.ROOT / ".ninjars-build/tmp") as directory:
            path = Path(directory) / "stress.json"
            path.write_text("{}")
            inputs = {}
            with patch.dict(sys.modules, {"ninjars_stress_v2": SimpleNamespace(report_snapshot=read)}), \
                    patch.object(report, "supplement_expectations", return_value=self.expected), \
                    patch.object(report, "candidate_runtime_signature", return_value=self.signature):
                result = report.stress_snapshot(path, {}, [], inputs)
            self.assertEqual(seen, [self.signature])
            self.assertTrue(result["complete_campaign_passed"])
            self.assertNotIn("performance", result["cases"][0])
            self.assertEqual(inputs[report.policy.relative(path)], report.workers.digest(path))

    def test_stress_rejects_relabelled_or_contradictory_results(self):
        valid = dict(self.expected, schema="madloop-final-stress-v1", status="passed",
                     complete_campaign_passed=True, cases=[])
        module = SimpleNamespace(report_snapshot=lambda *args, **kwargs: valid)
        with patch.dict(sys.modules, {"ninjars_stress_v2": module}), \
                patch.object(report, "supplement_expectations", return_value=self.expected), \
                patch.object(report, "candidate_runtime_signature", return_value=self.signature):
            for key in self.expected:
                changed = dict(valid, **{key: "other"})
                module.report_snapshot = lambda *args, **kwargs: changed
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "differently bound"):
                    report.stress_snapshot("unused.json", {}, [], {})
            module.report_snapshot = lambda *args, **kwargs: dict(valid, status="failed")
            with self.assertRaisesRegex(ValueError, "Contradictory"):
                report.stress_snapshot("unused.json", {}, [], {})
            module.report_snapshot = lambda *args, **kwargs: dict(valid, complete_campaign_passed="false")
            with self.assertRaisesRegex(ValueError, "Invalid"):
                report.stress_snapshot("unused.json", {}, [], {})


class NinjaRSRuntimeProvenanceReportTest(unittest.TestCase):
    def setUp(self):
        (report.ROOT / ".ninjars-build/tmp").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=report.ROOT / ".ninjars-build/tmp",
                                                prefix="runtime-report-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.revisions = {"ninjars": "a"*40, "oneloopmaster": "b"*40, "symbolica": "c"*40}
        self.config = self.write("ninjars-config.json", {"abi_version": 3, "local_sources": {}})
        self.cargo = self.root / "Cargo.toml"
        self.cargo.write_text('''[dependencies]
ninjars = { git = "https://example.org/ninjars", rev = "%s" }
ninjars-symbolica = { git = "https://example.org/ninjars", rev = "%s" }
[patch.crates-io]
symbolica = { git = "https://example.org/symbolica", rev = "%s" }
numerica = { git = "https://example.org/symbolica", rev = "%s" }
graphica = { git = "https://example.org/symbolica", rev = "%s" }
''' % (self.revisions["ninjars"], self.revisions["ninjars"],
       self.revisions["symbolica"], self.revisions["symbolica"], self.revisions["symbolica"]))
        self.lock = self.root / "Cargo.lock"
        self.lock.write_text("\n".join('''[[package]]
name = "%s"
version = "0.1.0"
source = "git+https://example.org/%s?rev=%s#%s"
''' % (crate, name, revision, revision) for name, crate, revision in (
            ("ninjars", "ninjars", self.revisions["ninjars"]),
            ("oneloopmaster", "oneloop", self.revisions["oneloopmaster"]),
            ("symbolica", "symbolica", self.revisions["symbolica"]))))
        self.archive = "d"*64
        self.installation = self.write("installation.json", {
            "passed": True, "exit_code": 0, "compiled_smoke_passed": True, "abi_version": 3,
            "revision": "source-commit", "files": {
                "Cargo.toml": report.workers.digest(self.cargo),
                "Cargo.lock": report.workers.digest(self.lock),
                "lib/ninjars-config.json": report.workers.digest(self.config),
                "lib/libninjars.a": self.archive}})
        self.item = {"role": "candidate_corrected", "backend": 8,
                     "generation_source_identity": {"tree_state_sha256": "generator"},
                     "runtime_metadata": {"ninjars-config.json": report.policy.bound_file(self.config)},
                     "libraries": {"libninjars.a": self.archive}}
        self.bindings = [{"targets": {"case-candidate-8": self.item}}]
        self.expected = {"core_dataset_identity_sha256": "core",
                         "literal_baseline_revision": report.BASELINE_REVISION,
                         "candidate_generator_identity_sha256": "generator",
                         "candidate_runtime_identity_sha256": "runtime"}
        self.manifest = dict(self.expected, schema="madloop-final-runtime-provenance-v1",
                             installation=report.policy.bound_file(self.installation),
                             cargo_toml=report.policy.bound_file(self.cargo),
                             cargo_lock=report.policy.bound_file(self.lock),
                             installed_config=report.policy.bound_file(self.config),
                             dependencies=self.revisions)
        self.path = self.write("manifest.json", report.policy.seal(self.manifest))

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def evaluate(self, specification=True):
        self.path.write_text(json.dumps(report.policy.seal(self.manifest)))
        with patch.object(report.policy, "reducer_signature", return_value={"identity_sha256": "runtime"}):
            return report.runtime_provenance_snapshot(self.path if specification else None,
                                                      {"identity_sha256": "core"}, self.bindings, {})

    def test_measured_abi3_metadata_is_distinct_from_historical_setup_provenance(self):
        pending = self.evaluate(False)
        self.assertEqual(pending["abi_version"], 3)
        self.assertEqual(pending["dependencies"], {"ninjars": None, "oneloopmaster": None, "symbolica": None})
        self.assertEqual(pending["status"], "pending")
        final = self.evaluate()
        self.assertEqual(final["status"], "passed")
        self.assertEqual(final["dependencies"], self.revisions)
        self.assertEqual(final["archive_sha256"], [self.archive])
        self.assertEqual(final["source_roles"]["candidate_corrected"],
                         [{"tree_state_sha256": "generator"}])

    def test_mismatched_dependency_or_candidate_identity_rejected(self):
        self.manifest["dependencies"] = dict(self.revisions, ninjars="e"*40)
        with self.assertRaisesRegex(ValueError, "revisions differ"):
            self.evaluate()
        self.manifest["dependencies"] = self.revisions
        self.manifest["candidate_runtime_identity_sha256"] = "older-runtime"
        with self.assertRaisesRegex(ValueError, "measured source/runtime"):
            self.evaluate()

    def test_unbound_archive_and_changed_installed_metadata_rejected(self):
        self.item["libraries"]["libninjars.a"] = "e"*64
        with self.assertRaisesRegex(ValueError, "measured archive"):
            self.evaluate()
        self.item["libraries"]["libninjars.a"] = self.archive
        self.lock.write_text(self.lock.read_text().replace("b"*40, "f"*40))
        with self.assertRaisesRegex(RuntimeError, "Changed bound evidence"):
            self.evaluate()

    def test_old_abi_and_unrecorded_local_overrides_cannot_certify_final_runtime(self):
        self.config.write_text(json.dumps({"abi_version": 2, "local_sources": {}}))
        self.item["runtime_metadata"]["ninjars-config.json"] = report.policy.bound_file(self.config)
        with self.assertRaisesRegex(ValueError, "ABI 3"):
            self.evaluate(False)
        self.config.write_text(json.dumps({"abi_version": 3, "local_sources": {"ninjars": "/tmp/clone"}}))
        self.item["runtime_metadata"]["ninjars-config.json"] = report.policy.bound_file(self.config)
        self.manifest["installed_config"] = report.policy.bound_file(self.config)
        installation = json.loads(self.installation.read_text())
        installation["files"]["lib/ninjars-config.json"] = report.workers.digest(self.config)
        self.installation.write_text(json.dumps(installation))
        self.manifest["installation"] = report.policy.bound_file(self.installation)
        with self.assertRaisesRegex(ValueError, "local source overrides"):
            self.evaluate()


class NinjaRSRegressionReportTest(unittest.TestCase):
    def setUp(self):
        self.spec = report.REPORT / "policy-v2/final-regressions-v1/validation.json"
        self.binding = report.policy.read(report.REPORT / (
            "policy-v2/runtime-bindings-abi3-tensor-factor-v1/"
            "representative-candidate-abi3-tensor-factor-v1-ggh-20261020.json"))

    def test_historical_reviewed_suites_keep_their_recorded_legacy_skip(self):
        # Check the retained evidence without claiming that its old source
        # hashes qualify the current implementation.
        loader = importlib.util.spec_from_file_location(
            "ninjars_test_historical_regression_seal", self.spec.with_name("verify_seal.py"))
        seal = importlib.util.module_from_spec(loader)
        with patch.object(sys, "dont_write_bytecode", True):
            loader.loader.exec_module(seal)
            self.assertEqual(seal.verify()["status"], "verified")
        spec = report.policy.read(self.spec)
        self.assertEqual((spec["tests_run"], spec["tests_passed"], spec["total_skipped"]), (90, 89, 1))
        self.assertEqual(sum(suite["skipped"] for suite in spec["suites"]), 1)
        for suite in spec["suites"]:
            log = report.ROOT / suite["logbound"]["path"]
            self.assertEqual(report.workers.digest(log), suite["logbound"]["sha256"])
            self.assertRegex(log.read_text(), rf"(?m)^Ran {suite['tests_run']} tests? in ")

    def test_historical_source_cannot_qualify_the_changed_current_generator(self):
        spec = report.policy.read(self.spec)
        filename = "madgraph/interface/madgraph_interface.py"
        self.assertNotEqual(report.workers.digest(report.ROOT / filename),
                            spec["production_source_sha256"][filename])
        with self.assertRaisesRegex(AssertionError, r"^madgraph/interface/madgraph_interface\.py$"):
            report.regression_snapshot(self.spec, [self.binding], {})


if __name__ == "__main__":
    unittest.main()
