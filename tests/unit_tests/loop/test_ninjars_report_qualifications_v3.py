"""Concrete retained qualification readers: no engines, builds or synthetic pass proofs."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools import ninjars_report_qualifications_v3 as qualification


class NinjaRSReportQualificationsV3Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=qualification.ROOT / ".ninjars-build/tmp",
                                                prefix="report-qualification-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def measured_candidate(self):
        filename = qualification.RELEASE / "generation/runtime-inventory.json"
        if not filename.is_file():
            self.skipTest("Actual final CLI runtime inventory is not retained in this checkout")
        case = json.loads(filename.read_text())["cases"]["qcd-1"]
        signature = json.loads((qualification.RELEASE / "generation/observed-qcd1-runtime.json").read_text())
        self.assertEqual(case["signatures"]["8"], signature["identity_sha256"])
        return signature, copy.deepcopy(
            case["generation_source_identity"]["tree_state"]["tracked"])

    def installation(self):
        if not (qualification.INSTALLATION.parent / "manifest.json").is_file():
            self.skipTest("Actual final installation/export capsule is not retained in this checkout")
        return self.measured_candidate()

    def test_unrelated_success_json_cannot_qualify_installation_or_software(self):
        filename = self.directory / "passed.json"
        filename.write_text(json.dumps({"passed": True, "status": "passed", "verified": True}))
        with self.assertRaisesRegex(ValueError, "concrete final installation"):
            qualification.installation_snapshot(filename, {}, {}, {})
        with self.assertRaisesRegex(ValueError, "concrete final software"):
            qualification.regression_snapshot(filename, {}, {})

    def test_actual_installed_exports_keep_their_specific_scopes(self):
        signature, generator = self.installation()
        inputs = {}
        result = qualification.installation_snapshot(qualification.INSTALLATION, signature, generator, inputs)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["archive_sha256"], next(value for name, value in
            signature["libraries"].items() if Path(name).name == "libninjars.a"))
        self.assertEqual([result["standalone"][key]["return_code"] for key in ("dp", "qp", "stable")],
                         [110, 120, 218])
        self.assertEqual((result["nlo_poles_passed"], result["nlo_poles_tested"]), (20, 20))
        self.assertEqual(result["nlo_pole_tolerance"], 1e-5)
        self.assertIn("no shower", result["scope"])
        self.assertNotIn("nrs_core_validation_passed", result)
        self.assertNotIn("final_publication_passed", result)
        for name in ("manifest.json", "verify.py", "runtime-results.json", "nlo-compile.log"):
            self.assertIn(qualification.policy.relative(qualification.INSTALLATION.parent / name), inputs)

    def test_retained_installation_cannot_certify_a_stale_archive(self):
        signature, generator = self.installation()
        for name in signature["libraries"]:
            if Path(name).name == "libninjars.a":
                signature["libraries"][name] = "0" * 64
        signature = qualification.policy.seal(signature)
        with self.assertRaisesRegex(ValueError, "archive differs"):
            qualification.installation_snapshot(qualification.INSTALLATION, signature, generator, {})

    def test_retained_installation_cannot_certify_other_interface_metadata(self):
        signature, generator = self.installation()
        for name in signature["metadata"]:
            if Path(name).name == "mninjars.mod":
                signature["metadata"][name] = "0" * 64
        signature = qualification.policy.seal(signature)
        with self.assertRaisesRegex(ValueError, "interface/link metadata differs"):
            qualification.installation_snapshot(qualification.INSTALLATION, signature, generator, {})

    def test_retained_installation_requires_same_measured_generator(self):
        signature, generator = self.installation()
        name = "madgraph/interface/madgraph_interface.py"
        self.assertIn(name, generator)
        generator[name] = "0" * 64
        with self.assertRaisesRegex(ValueError, "measured candidate generator"):
            qualification.installation_snapshot(qualification.INSTALLATION, signature, generator, {})

    def test_missing_generator_scope_cannot_inherit_retained_pass(self):
        signature, _ = self.installation()
        with self.assertRaisesRegex(ValueError, "measured candidate generator"):
            qualification.installation_snapshot(qualification.INSTALLATION, signature, {}, {})

    def test_actual_software_proof_keeps_fresh_and_reused_counts_separate(self):
        if not qualification.REGRESSIONS.is_file():
            self.skipTest("Actual final software capsule is not retained in this checkout")
        _, generator = self.measured_candidate()
        inputs = {}
        result = qualification.regression_snapshot(qualification.REGRESSIONS, generator, inputs)
        self.assertEqual(result["status"], "passed")
        self.assertEqual((result["tests_run"], result["fresh_tests_passed"], result["skipped"]), (47, 47, 0))
        self.assertEqual((result["reused_tests_observed"], result["reused_tests_passed"],
                          result["reused_tests_skipped"]), (21, 20, 1))
        self.assertIn("not rerun", result["suites"][1]["scope"])
        self.assertIn("zero actual reduction-engine", result["scope"])
        self.assertNotIn("nrs_core_validation_passed", result)
        for name in ("focused.log", "verify.py", "verify_reuse.py", "cfg-test-equivalence.json"):
            self.assertIn(qualification.policy.relative(qualification.REGRESSIONS.parent / name), inputs)

    def test_software_proof_requires_same_measured_generation_sources(self):
        if not qualification.REGRESSIONS.is_file():
            self.skipTest("Actual final software capsule is not retained in this checkout")
        _, generator = self.measured_candidate()
        generator["madgraph/interface/madgraph_interface.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "measured candidate generator"):
            qualification.regression_snapshot(qualification.REGRESSIONS, generator, {})

    def test_actual_new_source_convergence_does_not_relabel_historical_failures(self):
        if not qualification.NUMERICAL.is_file():
            self.skipTest("Actual final numerical capsule is not retained in this checkout")
        signature, _ = self.installation()
        manifest = json.loads(qualification.NUMERICAL.read_text())
        installed = qualification.ROOT / manifest["installed_precision_portable"]["path"]
        inputs = {}
        result = qualification.numerical_snapshot(qualification.NUMERICAL, installed, signature, inputs)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["records_each_precision"], 1392)
        self.assertEqual(result["raw_records"], 2785)
        self.assertTrue(result["final_source_convergence_passed"])
        self.assertFalse(result["historical_bit_identity_test_passed"])
        self.assertEqual(result["original_rust_replay_exit_code"], 101)
        self.assertEqual(result["historical_binary128_changed_records"], 468)
        self.assertFalse(result["known_provider_regression_test_passed"])
        self.assertEqual(result["known_provider_unsafe_successes"], 2)
        self.assertFalse(result["performance_claim"])
        self.assertIn("tools/ninjars_final_numerical.py", inputs)

    def test_new_source_controls_require_their_installed_abi_view(self):
        if not qualification.NUMERICAL.is_file():
            self.skipTest("Actual final numerical capsule is not retained in this checkout")
        signature, _ = self.installation()
        with self.assertRaisesRegex(ValueError, "actual installed64"):
            qualification.numerical_snapshot(qualification.NUMERICAL, None, signature, {})
        with self.assertRaisesRegex(ValueError, "actual installed64"):
            qualification.numerical_snapshot(qualification.NUMERICAL, self.directory / "passed.json", signature, {})

    def test_original_numerical_view_cannot_skip_relocation_authority(self):
        with self.assertRaisesRegex(ValueError, "concrete final numerical"):
            qualification.numerical_snapshot(qualification.NUMERICAL.parent / "manifest.json", None, {}, {})

    def test_optimized_python_cannot_skip_owner_verifier_assertions(self):
        command = [sys.executable, "-B", "-O", "-c",
            "import sys; sys.path.insert(0, 'tools'); import ninjars_report_qualifications_v3 as q; "
            "q.capsule(q.RELEASE / 'install-export', {}, lambda module: module.verify(live=False))"]
        observed = subprocess.run(command, cwd=qualification.ROOT, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(observed.returncode, 0)
        self.assertIn("assertions enabled", observed.stderr)


if __name__ == "__main__":
    unittest.main()
