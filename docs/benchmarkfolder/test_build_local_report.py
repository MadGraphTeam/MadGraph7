"""Local report evidence and presentation tests; no native measurements."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("build_local_report", HERE / "build_local_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value, sort_keys=True) + "\n")


class TestLocalReport(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.measurement = self.root / "local-1"
        self.private_report = self.measurement / "report"
        self.private_report.mkdir(parents=True)
        self.proof = {"numerical": "reparsed and sealed"}
        self.config = {"points_per_case": 100, "repetitions": 7, "minimum_batch_seconds": 1.,
                       "performance_upper_ratio": 1.05, "relative_tolerance": 3e-6,
                       "backends": list(report.BACKENDS),
                       "cases": [{"id": "example", "role": "main", "process": "u u~ > z"},
                                 {"id": "control", "role": "control", "process": "g g > h"}]}
        self.core = {"date": "2026-11-10", "source_date_epoch": 1640000000,
                     "gate": {"consistent_candidate_source": True}, "cases": []}
        for case in self.config["cases"]:
            self.core["cases"].append({"id": case["id"], "validation_errors": {},
                "correctness": [{"backend": backend, "status": "pending", "passed": None,
                                  "samples": 0, "max_relative_error": None,
                                  "ordinary_target_passed": None, "failures": [],
                                  "actual_backend_precision_counts": {}, "native_dp_only": None}
                                 for backend in report.BACKENDS]})
            for source, backend in [("baseline", 1), *(("candidate", b) for b in report.BACKENDS)]:
                write(self.private_report / "experiments/numerical" / ("ordinary-" + case["id"])
                      / f"{source}-{backend}/result.json",
                      {"status": "complete", "comparison": {"passed": True,
                       "ordinary_target_passed": backend != 6, "maximum_relative_error": 2e-8 if backend == 6 else 0.},
                       "observation": {"backend_precision_counts": {f"{backend}:DP": 100}, "cold_seconds": 1.25}})
        patch_path = report.bench.ROOT / "vendor/tirust/patches/manifest.json"
        self.value = {"cpu": 9, "source_identity": {"revision": "a" * 40, "files_sha256": "source-files",
                      "files": {"vendor/tirust/patches/manifest.json": report.bench.digest(patch_path)}},
                      "origin": {"campaign": "canonical", "manifest_sha256": "manifest",
                                 "manifest": {"candidate": {"numerical_identity_sha256": "arithmetic"}},
                                 "qualified_failures": []},
                      "native_recipe": {"manifest_sha256": "manifest", "lock_sha256": "lock", "git_dependencies": [{"name": "oneloop", "version": "0.1.0", "source": "git+https://example.invalid#" + report.ONELOOP_REVISION}]},
                      "native_source_binding": {"kind": "source-receipt"},
                      "local_deployment": {"native": {"files": {"lib/libmtirust.so": "native-bytes"}}},
                      "registry_identity": {"host": {"node": "fixture"}}}
        write(self.measurement / "measurement.json", self.value)
        write(self.measurement / "numerical-prerequisites.json", self.proof)
        write(self.measurement / "origin/cases.json", self.config)
        write(self.private_report / "inputs/candidate.json", {"source": "frozen"})
        write(self.private_report / "points/example.dat", "100\nfrozen fixture\n")
        self.mock(report.local, "roots", lambda measurement: (self.measurement, self.root / "work"))
        self.mock(report.bench, "REPORT", self.private_report)
        self.mock(report.bench, "WORK", self.root / "work")
        self.mock(report.local, "isolated", self.isolated)
        self.verify = self.mock(report.local, "verify_numerical", return_value=self.proof)
        self.mock(report.bench, "load_config", return_value=self.config)
        self.mock(report.bench, "summarize", side_effect=self.summarize)
        self.guard = self.mock(report.bench, "campaign_guard")
        self.results = {}
        self.validation = self.mock(report.bench, "validate_retained_result", side_effect=self.validated)

    def mock(self, obj, name, *args, **kwargs):
        patcher = patch.object(obj, name, *args, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    @contextlib.contextmanager
    def isolated(self, measurement):
        self.assertEqual(measurement, "local-1")
        yield self.value

    def summarize(self, config):
        write(self.private_report / "summary.json", self.core)
        return self.core

    def validated(self, directory, label, case, config):
        result = self.results[str(directory)]
        if isinstance(result, Exception):
            raise result
        self.assertEqual(result["label"], label)
        return result, {"unused": "retained validation anchor"}

    def result(self, case, label, ratio=.99, eligible=True, numerical=True, suffix=""):
        directory = self.private_report / "runs" / case / (label + suffix)
        value = {"label": label, "acceptance_eligible": eligible, "numerical_pass": numerical,
                 "passed": eligible and numerical and ratio <= 1.05,
                 "statistics": {"geometric_mean_ratio": ratio, "lower_95": ratio-.01,
                                "upper_95": ratio+.01, "pairs": 7},
                 "timings": {"baseline": [.01]*7, "candidate": [.01*ratio]*7},
                 "raw_observations": {"baseline": [{"backend_precision_counts": {"8:DP": 100}}]*7,
                                      "candidate": [{"backend_precision_counts": {"9:DP": 99, "9:QP": 1}}]*7}}
        write(directory / "summary.json", value)
        self.results[str(directory)] = value
        return directory

    def all_timings(self):
        for case in self.config["cases"]:
            for label in report.LABELS:
                self.result(case["id"], label)
        for case in self.core["cases"]:
            for check in case["correctness"]:
                check.update(status="complete", passed=True, samples=7)

    def test_data_only_keeps_pending_timing_and_target_misses(self):
        old = self.measurement / "origin/old/runs/example/tirust-first/summary.json"
        write(old, {"acceptance_eligible": True, "passed": False,
                    "statistics": {"geometric_mean_ratio": 1.2, "lower_95": 1.19, "upper_95": 1.21}})
        self.value["origin"]["qualified_failures"].append({"report": "old", "predecessor_id": "original",
            "failure": {"case": "example", "label": "tirust-first", "directory": "runs/example/tirust-first",
                        "files": {"summary.json": report.bench.digest(old)}}})
        write(self.measurement / "measurement.json", self.value)
        original_hash = report.bench.digest(old)
        output = report.build("local-1", "render-1", data_only=True)
        destination = Path(output["directory"])
        dossier, ledger = report.read(destination / "dossier.json"), report.read(destination / "build.json")
        self.assertEqual(output["status"], "data-only")
        self.assertIsNone(output["pdf"])
        self.assertFalse(output["overall_goal_qualified"])
        self.assertEqual(dossier["numerical"]["status"], "complete")
        self.assertFalse(dossier["numerical"]["performance_eligible"])
        self.assertFalse(dossier["numerical"]["cases"][0]["ordinary_target_passed"])
        self.assertTrue(dossier["numerical"]["cases"][0]["regression_pass"])
        self.assertEqual(dossier["timing"]["main_gate"]["status"], "pending")
        self.assertIsNone(dossier["timing"]["cases"][0]["comparisons"][0]["ratio"])
        self.assertEqual(dossier["historical_failures"][0]["ratio"], 1.2)
        self.assertEqual(report.bench.digest(old), original_hash)
        self.assertEqual(ledger["inputs"]["origin/old/runs/example/tirust-first/summary.json"], report.record(old))
        self.assertEqual(ledger["protocols"]["tirust_benchmark_local.py"], report.record(report.bench.ROOT / "tools/tirust_benchmark_local.py"))
        self.verify.assert_called_once_with(9)
        self.guard.assert_called_once_with("local-report-after")
        with self.assertRaisesRegex(ValueError, "append-only"):
            report.build("local-1", "render-1", data_only=True)

    def test_requires_reparsed_numerical_seal(self):
        self.verify.return_value = {"numerical": "changed"}
        with self.assertRaisesRegex(ValueError, "seal"):
            report.build("local-1", "render-1", data_only=True)
        self.assertEqual(report.read(self.measurement / "render-1/build.json")["status"], "failed")
        self.assertFalse((self.measurement / "render-1/dossier.json").exists())

    def test_controls_and_tirust_speed_remain_descriptive(self):
        self.all_timings()
        for label in report.LABELS:
            self.result("control", label, ratio=20.)
        self.result("example", "backend-9", ratio=3.)
        rows = report.timing_rows(self.config, self.core)
        self.assertTrue(rows["main_gate"]["passed"])
        self.assertTrue(rows["main_gate"]["complete"])
        descriptive = rows["cases"][0]["comparisons"][-1]
        self.assertEqual(descriptive["status"], "descriptive")
        self.assertIsNone(descriptive["passed"])
        self.assertFalse(descriptive["core_passed"])
        self.assertEqual(descriptive["usage"]["candidate"]["9:QP"], 7)

    def test_backend9_duration_is_required_despite_descriptive_ratio(self):
        self.all_timings()
        self.result("example", "backend-9", eligible=False)
        gate = report.timing_rows(self.config, self.core)["main_gate"]
        self.assertFalse(gate["complete"])
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["status"], "failed")

    def test_timed_standalone_correctness_is_separate_requirement(self):
        self.all_timings()
        self.core["cases"][0]["correctness"][-1].update(status="complete", passed=False)
        gate = report.timing_rows(self.config, self.core)["main_gate"]
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["status"], "failed")

    def test_inconsistent_source_cannot_pass_complete_timings(self):
        self.all_timings()
        self.core["gate"]["consistent_candidate_source"] = False
        gate = report.timing_rows(self.config, self.core)["main_gate"]
        self.assertTrue(gate["complete"])
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["status"], "failed")

    def test_unselected_qualified_failure_cannot_disappear(self):
        self.all_timings()
        self.result("example", "tirust-first", ratio=1.2)
        self.result("example", "tirust-first", suffix="-attempt-2")
        write(self.private_report / "runs/example/selected-attempts.json",
              {"tirust-first": {"directory": "tirust-first-attempt-2"}})
        rows = report.timing_rows(self.config, self.core)
        self.assertEqual(rows["cases"][0]["comparisons"][0]["status"], "passed")
        self.assertEqual(rows["main_gate"]["status"], "failed")
        self.assertEqual(len([row for row in rows["attempts"] if row["case"] == "example" and row["label"] == "tirust-first"]), 2)

    def test_failed_invalid_and_raw_only_attempts_are_preserved(self):
        write(self.private_report / "runs/example/backend-1/failure.json", {"error": "resource unavailable"})
        write(self.private_report / "runs/example/backend-6/raw.dat", "incomplete attempt\n")
        invalid = self.result("example", "backend-7")
        self.results[str(invalid)] = ValueError("raw evidence mismatch")
        attempts = report.timing_rows(self.config, self.core)["attempts"]
        self.assertEqual([item["status"] for item in attempts], ["failed", "pending", "invalid-evidence"])
        self.assertTrue(all(item["ratio"] is None for item in attempts))
        self.assertTrue(all(item["candidate_seconds_per_point"] is None for item in attempts))

    def test_cross_label_validation_failure_overrides_selection(self):
        self.all_timings()
        self.core["cases"][0]["validation_errors"]["backend-9"] = "cohort mismatch"
        rows = report.timing_rows(self.config, self.core)
        selected = rows["cases"][0]["comparisons"][-1]
        self.assertEqual(selected["status"], "invalid-evidence")
        self.assertIsNone(selected["ratio"])
        self.assertFalse(rows["main_gate"]["complete"])

    def test_missing_full_oneloop_pin_rejected(self):
        self.value["native_recipe"]["git_dependencies"][0]["source"] = "git+https://example.invalid#bd703d09"
        with self.assertRaisesRegex(ValueError, "oneloopmaster Git revision"):
            report.build("local-1", "render-1", data_only=True)

    def test_rejects_unsafe_render_label_and_nonfinite_input(self):
        for label in ("render-0", "../render-1", "latest", "render-1/child"):
            with self.assertRaises(ValueError):
                report.build("local-1", label, data_only=True)
        write(self.root / "nan.json", '{"ratio": NaN}')
        with self.assertRaisesRegex(ValueError, "Nonfinite JSON"):
            report.read(self.root / "nan.json")

    def test_actual_cli_data_only_route(self):
        output = io.StringIO()
        with patch.object(report.sys, "argv", [str(HERE / "build_local_report.py"), "--measurement", "local-1", "--render-label", "render-1", "--data-only"]), contextlib.redirect_stdout(output):
            report.main()
        self.assertEqual(json.loads(output.getvalue())["status"], "data-only")

    def test_font_inventory_is_explicit_and_content_bound(self):
        with self.assertRaisesRegex(ValueError, "TYPST_FONT_PATHS"):
            report.fonts({})
        font = self.root / "fonts/example.otf"
        write(font, "fixture font bytes")
        self.assertEqual(report.fonts({"TYPST_FONT_PATHS": str(font.parent)}), {str(font): report.record(font)})

    @unittest.skipUnless(shutil.which("typst") and os.environ.get("TYPST_FONT_PATHS"), "source tools/tirust_env.sh for actual document rendering")
    def test_actual_cli_pdf_with_pending_and_failed_fixture(self):
        write(self.private_report / "runs/example/backend-1/failure.json", {"error": "fixture resource failure"})
        write(self.private_report / "runs/example/backend-6/raw.dat", "fixture partial output\n")
        self.result("example", "backend-7", eligible=False)
        self.result("control", "backend-9", ratio=3.)
        output = io.StringIO()
        with patch.object(report.sys, "argv", [str(HERE / "build_local_report.py"), "--measurement", "local-1", "--render-label", "render-1"]), contextlib.redirect_stdout(output):
            report.main()
        result = json.loads(output.getvalue())
        destination = Path(result["directory"])
        ledger = report.read(destination / "build.json")
        self.assertEqual(result["status"], "rendered")
        self.assertEqual(Path(result["pdf"]), destination / "benchmark.pdf")
        self.assertTrue(Path(result["pdf"]).read_bytes().startswith(b"%PDF-"))
        self.assertEqual(ledger["pdf"], report.record(result["pdf"]))
        self.assertEqual(ledger["document_inputs"]["local-report.typ"], report.record(HERE / "local-report.typ"))
        self.assertIn("--ignore-system-fonts", ledger["typst"]["command"])
        self.assertEqual((destination / "typst.log").read_text(), "")


if __name__ == "__main__":
    unittest.main()
