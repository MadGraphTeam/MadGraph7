"""Final installed-archive precision proof must follow raw calls and inputs."""
import copy
from decimal import Decimal
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_precision_v2 as precision


class TestNinjaRSInstalledPrecision(unittest.TestCase):
    def test_old_archive_cannot_be_relabelled_with_final_runtime_identity(self):
        metadata = {name: name for name in precision.policy.RUNTIME_METADATA_NAMES | {"mninjars.mod"}}
        signature = precision.policy.seal({"schema": "reducer-runtime-signature-v2",
                                          "libraries": {"ninjars_lib/libninjars.a": "actual-archive"},
                                          "metadata": metadata, "link_inputs": {}})
        identity = signature["identity_sha256"]
        precision.match_runtime_archive(signature, "actual-archive", metadata, identity)
        with self.assertRaises(ValueError):
            precision.match_runtime_archive(signature, "stale-archive", metadata, identity)
        with self.assertRaises(ValueError):
            precision.match_runtime_archive(signature, "actual-archive", metadata, "relabelled")
        with self.assertRaises(ValueError):
            precision.match_runtime_archive(signature, "actual-archive", {**metadata, "mninjars.mod": "stale-module"}, identity)
        forged = copy.deepcopy(signature)
        forged["libraries"]["ninjars_lib/libninjars.a"] = "stale-archive"
        with self.assertRaises(RuntimeError):
            precision.match_runtime_archive(forged, "stale-archive", metadata, identity)

    def test_all_seven_fixtures_all_three_modes_are_included(self):
        requests = precision.fixture_requests(ROOT / "vendor/ninjars/tests/data")
        self.assertEqual(len(requests), 63)
        self.assertEqual(len({r["case"] for r in requests}), 7)
        self.assertEqual(len({r["lane"] for r in requests}), 21)
        self.assertEqual({r["mode"] for r in requests}, {0, 1, 2})
        for row in requests:
            self.assertEqual(len(row["s_matrix"]), row["n"]**2)
            self.assertEqual(len(row["momenta"]), 4*row["n"])
            self.assertEqual(len(row["masses_squared"]), 2*row["n"])

    def test_convergence_uses_more_than_binary64_information(self):
        reference = ["1.0000000000000000000000000000000001"] + ["0"]*7
        exact = precision.verify_convergence(reference, reference)
        self.assertTrue(exact["passed"])
        rounded = precision.verify_convergence(["1"]+["0"]*7, reference)
        self.assertNotEqual(rounded["maximum_scaled_error"], "0")
        bad = precision.verify_convergence(["1.000000000000000000000000000001"]+["0"]*7, ["1"]+["0"]*7)
        self.assertFalse(bad["passed"])

    def test_nonfinite_zero_or_missing_convergence_is_rejected(self):
        for left, right in ((["0"]*8, ["0"]*8), (["NaN"]+["0"]*7, ["1"]+["0"]*7),
                            (["1"], ["1"]*8)):
            with self.subTest(left=left), self.assertRaises(ValueError):
                precision.verify_convergence(left, right)

    def test_large_rational_cancellation_cannot_mask_laurent_error(self):
        reference = ["1"]+["0"]*5+["1e100", "0"]
        altered = ["1.0000000001"]+["0"]*5+["1e100", "0"]
        result = precision.verify_convergence(altered, reference)
        self.assertFalse(result["passed"])
        self.assertEqual(Decimal(result["maximum_scaled_error"]), Decimal("1e-10"))
        self.assertEqual(Decimal(result["rational_maximum_scaled_difference"]), 0)

    @staticmethod
    def request(mode=2):
        return {"record": 0, "case": "case", "lane": "lane", "mode": mode,
                "reference": {k: [0., 0.] for k in ("eps0", "eps_m1", "eps_m2", "rational")}}

    def test_basis_dependent_rational_split_is_reported_separately(self):
        actual = {"record": 0, "mode": 2, "status": 0, "output": [0.]*6+[1., 0.]}
        result = precision.compare_requests([self.request()], [actual])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["checks"][0]["basis_dependent_rational_scaled_difference"], 1.)
        self.assertFalse(result["checks"][0]["rational_compared"])
        actual["mode"] = 0
        result = precision.compare_requests([self.request(0)], [actual])
        self.assertEqual(result["status"], "failed")

    def test_status_and_order_failures_are_never_hidden(self):
        request = self.request()
        failed = {"record": 0, "mode": 2, "status": 1, "output": ["!"]*8}
        self.assertEqual(precision.compare_requests([request], [failed])["status"], "failed")
        with self.assertRaises(ValueError):
            precision.compare_requests([request], [])
        failed["record"] = 1
        with self.assertRaises(ValueError):
            precision.compare_requests([request], [failed])

    def test_crashed_or_partial_campaign_retains_failed_observation(self):
        request = self.request()
        result = precision.execution_summary([request], "license failure\n", 101)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failures"][-1]["observed_calls"], 0)
        self.assertEqual(result["failures"][-1]["exit_code"], 101)
        malformed = precision.execution_summary([request], "NINJARS_INSTALLED {\n", None)
        self.assertEqual(malformed["status"], "failed")
        self.assertIsNotNone(malformed["failures"][-1]["parser_error"])

    def test_actual_installed_256_must_match_final_source_unit_256(self):
        request = {"record": 0, "case": "case", "lane": "lane", "mode": 2,
                   "reference_256": ["1"]+["0"]*7, "reference_384": ["1"]+["0"]*7}
        actual = {"record": 0, "mode": 2, "status": 0, "output": ["1"]+["0"]*7}
        self.assertEqual(precision.compare_requests([request], [actual])["status"], "passed")
        actual["output"][0] = "1.0000000000000000000000000000000001"
        # This still meets the convergence tolerance; exact same-source
        # production output identity catches it independently.
        self.assertTrue(precision.verify_convergence(actual["output"], request["reference_384"])["passed"])
        self.assertEqual(precision.compare_requests([request], [actual])["status"], "failed")


if __name__ == "__main__":
    unittest.main()
