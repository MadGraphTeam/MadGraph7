"""Report regression tests: missing, incomplete, stale and disagreeing evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools import ninjars_report as report


class NinjaRSReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ninjars-report-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.docs = self.root / "report"
        self.work = self.root / "work"
        self.config = {
            "points_per_case": 2, "repetitions": 7, "relative_tolerance": 1e-7,
            "scale_tolerance": 1e-10, "max_runtime_ratio": 1.05,
            "cases": [{"id": "qcd-1", "process": "u u~ > z [virt=QCD]",
                       "nfinal": 1, "model": "loop_sm"}],
        }
        self.write(self.docs / "cases.json", self.config)
        self.write(self.docs / "provenance.json", {})
        descriptor = {
            "schema": "fresh-process-v1",
            "sources": {name: hashlib.sha256(b"fixture").hexdigest()
                        for name in ("ninjars_benchmark.py", "ninjars_benchmark_driver.f")}}
        self.protocol_identity = report.benchmark.content_digest(descriptor)
        protocol = self.docs / "inputs/benchmark-protocols" / self.protocol_identity
        self.write(protocol.with_suffix(".json"), descriptor)
        protocol.mkdir()
        for name in descriptor["sources"]:
            (protocol / name).write_bytes(b"fixture")

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def snapshot(self):
        return report.build_snapshot(self.docs, self.work, timestamp="test")

    def run_record(self, source, backend, repetition):
        code = {1: 211, 6: 216, 7: 217, 8: 218}[backend]
        return {
            "source": source, "case": "qcd-1", "backend": backend,
            "benchmark_protocol": "fresh-process-v1",
            "protocol_identity_sha256": self.protocol_identity,
            "repetition": repetition, "seconds_per_point": backend * 1e-6,
            "cold_seconds": 0.5, "loops": 1, "cpu": 3,
            "timed_status_counts": {str(code): 2}, "status_counts": {str(code): 2},
            "points_sha256": "points", "parameter_card_sha256": "parameters",
            "paired_baseline_candidate": True, "measurement_run_id": "paired",
            "executable_sha256": source, "git_revision": "revision",
            "shared_numerical_settings": {"ctmoderun": -1, "mlstabthres": 1e-3},
            "rows": [{"point": p, "order": o, "status": code, "accuracy": 1e-12,
                      "coefficients": [1.0, 2.0, 3.0, 4.0]}
                     for p in (1, 2) for o in (0, 1)],
        }

    def populate(self, repetitions):
        for source, backend in [("baseline", 6)] + [("candidate", b) for b in report.BACKENDS]:
            for rep in range(repetitions):
                self.write(self.docs / "data" / f"{source}-qcd-1-{backend}-{rep}.json",
                           self.run_record(source, backend, rep))

    def test_missing_and_calibration_stay_pending(self):
        self.write(self.docs / "data/candidate-qcd-1-8-calibration.json",
                   self.run_record("candidate", 8, "calibration"))
        result = self.snapshot()
        self.assertEqual(result["summary"]["measured_candidate_cells"], 0)
        self.assertEqual(result["cases"][0]["agreement"]["status"], "pending")
        self.assertIsNone(result["cases"][0]["timings"]["candidate"]["8"]["median_us"])

    def test_partial_repetitions_never_become_final_timing(self):
        self.populate(1)
        result = self.snapshot()["cases"][0]
        self.assertEqual(result["agreement"]["status"], "passed")
        self.assertEqual(result["agreement"]["comparisons"], 96)
        self.assertEqual(result["timings"]["candidate"]["8"]["status"], "partial")
        self.assertIsNone(result["timings"]["candidate"]["8"]["median_us"])
        self.assertEqual(result["performance"]["status"], "pending")

    def test_complete_medians_and_precision_counters(self):
        self.populate(7)
        for rep in range(7):
            record = self.run_record("candidate", 8, rep)
            record["timed_status_counts"] = {"218": 1, "322": 1}
            self.write(self.docs / "data" / f"candidate-qcd-1-8-{rep}.json", record)
        result = self.snapshot()
        self.assertEqual(result["summary"]["measured_candidate_cells"], 4)
        measured = result["cases"][0]["timings"]["candidate"]["8"]
        self.assertAlmostEqual(measured["median_us"], 8)
        self.assertAlmostEqual(measured["ratio"], 8 / 6)
        self.assertEqual((measured["dp"], measured["qp"]), (7, 7))

    def test_disagreement_is_recomputed_from_all_coefficients(self):
        self.populate(1)
        record = self.run_record("candidate", 8, 0)
        record["rows"][3]["coefficients"][3] = 10.0
        self.write(self.docs / "data/candidate-qcd-1-8-0.json", record)
        measured = self.snapshot()["cases"][0]["agreement"]
        self.assertEqual(measured["status"], "failed")
        self.assertEqual(measured["failures"], 3)
        self.assertGreater(measured["max_tolerance_fraction"], 1)

    def test_stale_performance_summary_is_rejected(self):
        self.populate(7)
        self.write(self.docs / "data/performance-qcd-1.json",
                   {"measurements": {"baseline": {"samples": [123.0] * 7}}})
        with self.assertRaisesRegex(ValueError, "Stale performance summary"):
            self.snapshot()

    def test_mixed_inputs_and_wrong_backend_are_rejected(self):
        self.populate(1)
        record = self.run_record("candidate", 8, 0)
        changed = copy.deepcopy(record)
        changed["points_sha256"] = "different"
        self.write(self.docs / "data/candidate-qcd-1-8-0.json", changed)
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            self.snapshot()
        changed = copy.deepcopy(record)
        changed["timed_status_counts"] = {"216": 2}
        self.write(self.docs / "data/candidate-qcd-1-8-0.json", changed)
        with self.assertRaisesRegex(ValueError, "Unexpected reducer"):
            self.snapshot()

    def test_every_backend_rejects_mixed_executables_and_settings(self):
        self.populate(7)
        for backend in report.BACKENDS:
            for key, changed in (
                    ("executable_sha256", "rebuilt-executable"),
                    ("shared_numerical_settings", {"ctmoderun": 1}),
                    ("cpu", 9)):
                with self.subTest(backend=backend, key=key):
                    path = self.docs / "data" / f"candidate-qcd-1-{backend}-6.json"
                    original = self.run_record("candidate", backend, 6)
                    record = copy.deepcopy(original)
                    record[key] = changed
                    self.write(path, record)
                    with self.assertRaisesRegex(ValueError, "Mixed|Different pinned"):
                        self.snapshot()
                    self.write(path, original)

    def test_completed_timings_require_recorded_settings(self):
        self.populate(7)
        for rep in range(7):
            record = self.run_record("candidate", 7, rep)
            del record["shared_numerical_settings"]
            self.write(self.docs / "data" / f"candidate-qcd-1-7-{rep}.json", record)
        with self.assertRaisesRegex(ValueError, "Missing shared_numerical_settings"):
            self.snapshot()

    def test_mixed_timing_protocol_or_validation_is_rejected(self):
        self.populate(7)
        path = self.docs / "data/candidate-qcd-1-8-6.json"
        for field, value in (("benchmark_protocol", "persistent-worker-v1"),
                             ("protocol_identity_sha256", "0"*64),
                             ("protocol_validation_sha256", "1"*64)):
            changed = self.run_record("candidate", 8, 6)
            changed[field] = value
            self.write(path, changed)
            with self.assertRaisesRegex(ValueError, "Mixed benchmark protocols"):
                self.snapshot()

    def test_unlabelled_legacy_probes_cannot_become_final_timings(self):
        self.populate(7)
        for rep in range(7):
            path = self.docs / "data" / f"candidate-qcd-1-8-{rep}.json"
            changed = self.run_record("candidate", 8, rep)
            del changed["benchmark_protocol"], changed["protocol_identity_sha256"]
            self.write(path, changed)
        with self.assertRaisesRegex(ValueError, "unsupported benchmark_protocol"):
            self.snapshot()

    def test_persistent_timings_require_complete_promotion_evidence(self):
        self.populate(7)
        for rep in range(7):
            path = self.docs / "data" / f"candidate-qcd-1-8-{rep}.json"
            changed = self.run_record("candidate", 8, rep)
            changed.update(benchmark_protocol="persistent-worker-v1",
                           protocol_validation_sha256="1"*64,
                           protocol_run_manifest_sha256="2"*64)
            self.write(path, changed)
        with self.assertRaisesRegex(ValueError, "complete validated repetition"):
            self.snapshot()

    def test_retained_preparation_is_used_without_disposable_outputs(self):
        retained = {"seconds": 18.2, "build_kind": "clean-subprocess",
                    "preexisting_subprocess_objects": 0}
        self.write(self.docs / "inputs/baseline/qcd-1/compilation.json", retained)
        # Retained metadata remains authoritative if a later experiment changes
        # the disposable build tree or the tree is removed altogether.
        path = self.work / "outputs/baseline/qcd-1/compilation.json"
        self.write(path, {"seconds": 0.1, "build_kind": "incremental"})
        self.assertEqual(
            self.snapshot()["cases"][0]["preparation"]["baseline"]["compilation"],
            retained)
        path.unlink()
        self.assertEqual(
            self.snapshot()["cases"][0]["preparation"]["baseline"]["compilation"],
            retained)
