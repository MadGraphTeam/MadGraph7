"""Numerical and evidence boundaries of the fixed final supplement."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_supplement_v2 as supplement


class TestNinjaRSSupplement(unittest.TestCase):
    def test_threshold_has_exact_declared_cartesian_product(self):
        points, grid = supplement.threshold_points(173., 125.)
        self.assertEqual(len(points), 12)
        plan = supplement.frozen_configuration()["gg_hh_threshold"]
        self.assertEqual([(g["delta"], g["cos_theta"]) for g in grid],
                         [(d, c) for d in plan["delta"] for c in plan["cos_theta"]])
        with tempfile.TemporaryDirectory(dir=ROOT / ".benchmark-work") as directory:
            path = Path(directory) / "points.dat"
            path.write_text("12\n" + "".join(" ".join(f"{x:.17e}" for x in p)+"\n"
                                           for event in points for p in event))
            result = supplement.point_metadata(path, [0., 0., 125., 125.], 2)
        self.assertLess(result["max_relative_conservation_residual"], 1e-14)
        self.assertLess(result["max_relative_mass_shell_residual"], 1e-14)

    def test_incompatible_threshold_card_is_rejected(self):
        for top, higgs in ((125., 173.), (0., 125.), (math.nan, 125.)):
            with self.subTest(top=top, higgs=higgs), self.assertRaises(ValueError):
                supplement.threshold_points(top, higgs)

    def test_frozen_case_inventory_and_thresholds_cannot_be_changed(self):
        config = supplement.loop_configuration()
        supplement.validate_configuration(config)
        for key, value in (("points_per_case", 99), ("seed", 1),
                           ("repetitions", 6), ("relative_tolerance", 1e-6)):
            changed = copy.deepcopy(config)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                supplement.validate_configuration(changed)

    def test_nonfinite_or_nonconserving_points_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".benchmark-work") as directory:
            path = Path(directory) / "points.dat"
            for raw in ("1\n1 0 0 1\n1 0 0 -1\nnan 0 0 0\n",
                        "1\n1 0 0 1\n1 0 0 -1\n3 0 0 0\n"):
                path.write_text(raw)
                with self.assertRaises(ValueError):
                    supplement.point_metadata(path, [0., 0., 2.], 1)

    @staticmethod
    def row(value=1., accuracy=1e-14, status=218, point=1):
        return {"point": point, "order": 0, "coefficients": [0., value, 0., 0.],
                "accuracy": accuracy, "status": status}

    def test_estimate_and_observed_error_are_separate(self):
        groups = {"candidate_corrected/8": [self.row(1.01, 1e-30)],
                  "original_helper_corrected/6": [self.row(status=216)],
                  "original_helper_corrected/7": [self.row(status=217)]}
        result = supplement.stability_summary(groups)["targets"]["candidate_corrected/8"]
        self.assertEqual(result["estimated_accuracy"]["maximum"], 1e-30)
        self.assertFalse(result["observed_discrepancy"]["passed"])
        self.assertGreater(result["observed_discrepancy"]["max_tolerance_fraction"], 1.)
        self.assertEqual(result["dp"], 1)
        self.assertEqual(result["qp"], 0)

    def test_full_qp_status_is_counted_and_all_points_retained(self):
        groups = {"candidate_corrected/8": [self.row(status=322), self.row(point=2)],
                  "original_helper_corrected/6": [self.row(status=328), self.row(status=216, point=2)]}
        result = supplement.stability_summary(groups)["targets"]["candidate_corrected/8"]
        self.assertEqual((result["dp"], result["qp"], result["ninjars_genuine_qp"]), (1, 1, 1))
        self.assertEqual([r["point"] for r in result["per_point"]], [1, 2])
        self.assertEqual(result["per_point"][0]["H"], 3)
        self.assertEqual(result["per_point"][0]["T"], 2)
        self.assertEqual(result["per_point"][0]["U"], 2)

    def test_duplicate_or_missing_order_rows_are_rejected(self):
        row = self.row()
        for groups in ({"a": [row, row], "b": [row]}, {"a": [row], "b": []}):
            with self.assertRaises(ValueError):
                supplement.compare_rows(groups)

    def test_zero_scale_and_nonfinite_coefficients(self):
        row = self.row(0.)
        self.assertTrue(supplement.compare_rows({"a": [row], "b": [row]})["passed"])
        invalid = self.row(math.nan)
        with self.assertRaises(ValueError):
            supplement.compare_rows({"a": [row], "b": [invalid]})

    def test_accuracy_population_rejects_invalid_values(self):
        for values in ([], [-1.], [math.inf], [math.nan]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                supplement.distribution(values)

    def test_missing_measurements_cannot_close_supplement(self):
        expected = {"core_dataset_identity_sha256": "a"*64,
                    "literal_baseline_revision": supplement.bench.BASELINE_REVISION,
                    "candidate_generator_identity_sha256": "b"*64,
                    "candidate_runtime_identity_sha256": "c"*64}
        data = {"source_groups": [], "failed_groups": [],
                "precision_controls": {"status": "pending", "working_bits": 256,
                                       "convergence_bits": 384, "checks": []},
                "identity_sha256": "d"*64, **expected,
                "point_metadata": {"supplement-gg-h": {}, "supplement-gg-hh": {}}}
        with mock.patch.object(supplement, "validate_supplement", return_value=(data, [])), \
                mock.patch.object(supplement, "checked", return_value=ROOT / "docs/benchmarkfolder/policy-v2/controls.json"):
            snapshot = supplement.report_snapshot("unused", expected)
            self.assertFalse(snapshot["complete_campaign_passed"])
            self.assertEqual(snapshot["status"], "unresolved")
            self.assertTrue(all(c["agreement"]["status"] == "pending" for c in snapshot["cases"]))
            self.assertTrue(all(not c["born_verified_zero"] for c in snapshot["cases"]))
            with self.assertRaises(ValueError):
                supplement.report_snapshot("unused", {**expected, "candidate_runtime_identity_sha256": "e"*64})
            data["precision_controls"]["status"] = "passed"
            with self.assertRaises(ValueError):
                supplement.report_snapshot("unused", expected)


if __name__ == "__main__":
    unittest.main()
