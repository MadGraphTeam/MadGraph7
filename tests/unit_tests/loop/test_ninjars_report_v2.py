"""Independent report checks; structural source/worker validation has its own suite."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import ninjars_report_v2 as report


class NinjaRSReportV2Test(unittest.TestCase):
    def setUp(self):
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
        self.assertTrue((target / "benchmark.typ").exists())
        with self.assertRaisesRegex(ValueError, "exists"):
            report.render(self.dataset_path, target)

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

    def test_wrong_original_magnitude_does_not_relax_corrected_comparison(self):
        self.fixture(original_offset=1e20, reference_offset=.01)
        self.assertEqual(self.snapshot()["cases"][0]["agreement"]["status"], "failed")

    def test_changed_raw_file_is_rejected_after_dataset_verification(self):
        self.fixture()
        filename = next(iter(self.dataset["measurements_sha256"]))
        self.policy.path(filename).write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed after"):
            self.snapshot()


if __name__ == "__main__":
    unittest.main()
