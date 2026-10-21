"""Raw numerical publication rejects omitted data and false convergence claims."""
import copy
from fractions import Fraction as Q
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_final_numerical as numeric


class TestFinalNumericalEvidence(unittest.TestCase):
    @staticmethod
    def stream():
        original, low, high = [], [], []
        for index in range(1392):
            old = {"n": 2, "rank": 2, "record": index, "output": ["1"] + ["0"] * 5 + ["100", "0"]}
            raw_values = (1, 0, 0, 0, 0, 0, -1, 0, 2, 0)
            observed = {"status": 0, "working_bits": 256, "input_unchanged": True, "output_bytes": 512,
                        "output": ["1"] + ["0"] * 5 + ["2", "0"],
                        "raw": [{"finite": True, "numerator": str(v), "denominator": "1", "negative": v < 0,
                                 "precision": 200} for v in raw_values]}
            original.append(old)
            low.append({"kind": "replay", "sequence": index, "input": old, "observed": observed,
                        "historical_binary128_values_and_zero_signs_equal": False, "historical_ascii_equal": False})
            high.append({"kind": "convergence", "sequence": index, "input": old,
                         "observed_384": {**copy.deepcopy(observed), "working_bits": 384},
                         "comparison": {"passed": True, "sign_changed_indices": []}})
        footer = {"failed_binary128_replay_records": 1392, "historical_ascii_changed_records_retained": 1392,
                  "failed_convergence_records": 0, "sign_changed_convergence_records_require_review": 0}
        return original, low + high + [footer]

    def test_historical_R1_changes_do_not_replace_new_convergence(self):
        original, captured = self.stream()
        result = numeric.recompute(original, captured)
        self.assertTrue(result["final_source_convergence_passed"])
        self.assertFalse(result["historical_bit_identity_passed"])
        self.assertEqual(result["historical_binary128_changed_records"], 1392)
        self.assertEqual(Q(result["maximum_scaled_errors"]["old_laurent"]), 0)
        self.assertEqual(Q(result["maximum_scaled_errors"]["old_R1"]), 98)

    def test_filtered_reordered_and_failed_streams_are_rejected(self):
        for mutation in ("missing", "reordered", "failed", "partial", "nonfinite"):
            original, captured = self.stream()
            if mutation == "missing":
                captured.pop(0)
            elif mutation == "reordered":
                captured[0], captured[1] = captured[1], captured[0]
            elif mutation == "failed":
                captured[0]["observed"]["status"] = 1
            elif mutation == "partial":
                captured[0]["observed"]["output_bytes"] = 511
            else:
                captured[0]["observed"]["raw"][0]["finite"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                numeric.recompute(original, captured)

    def test_CC_R1_cancellation_cannot_hide_precision_instability(self):
        original, captured = self.stream()
        # The complete Laurent series stays identical, but its decomposition
        # changes between precisions. Both the decomposition and full value
        # must independently converge in a single source/basis.
        captured[1392]["observed_384"]["raw"][6]["numerator"] = "-2"
        captured[1392]["observed_384"]["raw"][8]["numerator"] = "3"
        with self.assertRaisesRegex(ValueError, "convergence failed for CC"):
            numeric.recompute(original, captured)

    def test_inconsistent_complete_finite_is_rejected(self):
        original, captured = self.stream()
        for index in (0, 1392):
            key = "observed" if index == 0 else "observed_384"
            captured[index][key]["raw"][6]["numerator"] = "-2"
        with self.assertRaisesRegex(ValueError, "not CC"):
            numeric.recompute(original, captured)

    def test_nearest_even_binary128_and_subnormal_rounding(self):
        unit = Q(1, 2**112)
        self.assertEqual(numeric.rounded113(1 + unit/2), 1)
        self.assertEqual(numeric.rounded113(1 + 3*unit/2), 1 + 2*unit)
        self.assertEqual(numeric.rounded113(-1 - unit/2), -1)
        self.assertEqual(numeric.rounded113(Q(1, 2**16495)), 0)
        self.assertEqual(numeric.rounded113(Q(3, 2**16495)), Q(1, 2**16493))
        self.assertEqual(numeric.ascii113("-0"), (0, True))

    def test_optimized_python_cannot_skip_failure_checks(self):
        code = "import sys;sys.path.insert(0,sys.argv[1]);import ninjars_final_numerical as n;n.require(False,'retained failure')"
        result = subprocess.run([sys.executable, "-O", "-c", code, str(ROOT / "tools")],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("retained failure", result.stderr)

    def test_metadata_relocation_cannot_change_physics_runtime_or_scope(self):
        original = {"stdout": {"path": "old", "sha256": "same"}, "raw": [1, 2, 3],
                    "runtime": "actual", "source": {"lib.rs": "tested"}, "scope": "actual64"}
        moved = {**copy.deepcopy(original), "stdout": {"path": "public", "sha256": "same"},
                 "original_metadata": {"path": "original-proof"}, "publication_relocation": "paths only"}
        numeric.relocation_delta(original, moved, ("stdout",), "test")
        for field, value in (("raw", [1, 2, 4]), ("runtime", "old-runtime"),
                             ("source", {"lib.rs": "unrelated"}), ("scope", "all processes")):
            altered = {**copy.deepcopy(moved), field: value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Metadata relocation changed"):
                numeric.relocation_delta(original, altered, ("stdout",), "test")

    def test_metadata_relocation_cannot_add_or_drop_unknown_fields(self):
        original = {"path": "old", "passed": True}
        for moved in ({"path": "new"}, {"path": "new", "passed": True, "invented": True}):
            with self.subTest(moved=moved), self.assertRaises(ValueError):
                numeric.relocation_delta(original, moved, ("path",), "test")


if __name__ == "__main__":
    unittest.main()
