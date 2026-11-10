import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import summarize as helper


def workers():
    result = []
    for backend, mode in helper.ARMS:
        for observed in (0, 1):
            result.append({"backend": backend, "mode": mode, "observed": observed,
                           "host_checked": False, "warnings": [], "public_raw": ["matching synthetic parity witness"],
                           "points": [{"point": p, "status": 120, "accuracy": [-1.0, -1.0], "host_checked": False,
                                       "coefficients": [1.0 + i for i in range(8)],
                                       "inputs": {"shared_input_sha256": "%064x" % p} if observed else None}
                                      for p in helper.POINTS]})
    return result


class SummaryTests(unittest.TestCase):
    def test_scalar_slot_order_and_pole_separation(self):
        source = workers()
        for row in source:
            if row["backend"] == 6:
                row["points"][0]["coefficients"][2] += 1
        answer = helper.summarize(source)
        self.assertEqual(answer["potential_scalar_comparisons"], 384)
        self.assertEqual(len(answer["pair_breakdown"]), 16)
        self.assertEqual([(s["sector"], s["coefficient"]) for s in answer["slots"]],
                         [(sector, name) for sector in ("total", "split_order_1") for name in ("Born", "finite", "single_pole", "double_pole")])
        self.assertEqual(answer["slot_summaries"][1]["metrics"]["symmetric"]["maximum"], 0)
        self.assertGreater(answer["slot_summaries"][2]["metrics"]["symmetric"]["maximum"], 0)
        self.assertEqual(answer["slot_summaries"][3]["metrics"]["symmetric"]["maximum"], 0)

    def test_zero_and_opposite_handling(self):
        self.assertEqual(helper.delta(0, 0), {"absolute": 0, "relative_to_left": 0, "symmetric": 0, "opposite_nonzero_sum": False})
        self.assertIsNone(helper.delta(0, 1)["relative_to_left"])
        self.assertEqual(helper.delta(0, 1)["symmetric"], 1)
        self.assertIsNone(helper.delta(1, -1)["symmetric"])
        self.assertTrue(helper.delta(1, -1)["opposite_nonzero_sum"])

    def test_undefined_symmetric_count_not_converted_to_zero(self):
        source = workers()
        for row in source:
            if row["backend"] == 6:
                row["points"][0]["coefficients"][1] = -2
        stats = helper.summarize(source)["slot_summaries"][1]["metrics"]["symmetric"]
        self.assertGreater(stats["undefined_count"], 0)
        self.assertEqual(stats["defined_count"] + stats["undefined_count"], 48)

    def test_shared_input_rejection_suppresses_every_numeric_summary(self):
        source = workers()
        next(w for w in source if w["backend"] == 9 and w["mode"] == 4 and w["observed"])["points"][0]["inputs"]["shared_input_sha256"] = "f" * 64
        answer = helper.summarize(source)
        self.assertTrue(answer["numerical_summary_suppressed"])
        self.assertIsNone(answer["slot_summaries"])
        self.assertEqual(answer["scalar_comparisons_reported"], 0)
        self.assertEqual(answer["cuttools_reference_rows"], [])

    def test_shape_and_duplicate_identity_rejected(self):
        for fault in ("workers", "points", "coefficients", "identity", "digest", "point_type", "point_object"):
            source = workers()
            if fault == "workers": source.pop()
            if fault == "points": source[0]["points"].pop()
            if fault == "coefficients": source[0]["points"][0]["coefficients"].pop()
            if fault == "identity": source[1] = copy.deepcopy(source[0])
            if fault == "digest": source[1]["points"][0]["inputs"]["shared_input_sha256"] = "not-a-digest"
            if fault == "point_type": source[0]["points"][0]["point"] = True
            if fault == "point_object": source[0]["points"][0] = None
            with self.subTest(fault=fault), self.assertRaises(ValueError): helper.summarize(source)

    def test_nonfinite_bool_and_finite_metric_overflow_rejected(self):
        for value in (float("nan"), float("inf"), True):
            source = workers(); source[0]["points"][0]["coefficients"][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError): helper.summarize(source)
        for left, right in ((1e308, 9e307), (1e308, -1e308), (5e-324, 1e308)):
            with self.subTest(pair=(left, right)), self.assertRaises(ValueError): helper.delta(left, right)

    def test_unchecked_status_and_accuracy_remain_mandatory(self):
        for field, value in (("status", 320), ("accuracy", [0, 0]), ("host_checked", True)):
            source = workers(); source[0]["points"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): helper.summarize(source)

    def test_public_parity_and_undefined_evidence(self):
        source = workers(); source[1]["public_raw"] = ["different"]
        with self.assertRaises(ValueError): helper.summarize(source)
        source = workers(); source[0].pop("public_raw")
        with self.assertRaises(ValueError): helper.summarize(source)

    def test_deterministic_first_maximum_witness(self):
        answer = helper.summarize(workers())
        witness = answer["slot_summaries"][0]["metrics"]["symmetric"]["maximum_witness"]
        self.assertEqual((witness["left"], witness["right"], witness["point"]), ([1, 4], [1, 5], 1))

    def test_combined_analysis_recomputes_every_supplied_delta(self):
        source = workers()
        keyed = {(w["backend"], w["mode"], w["observed"]): w for w in source}
        combined = {"schema": "qcd-n3-genuine-qp-diagnostic-v1", "host_checked": False,
                    "timing_or_campaign_acceptance": False, "all_shared_input_gates_pass": True,
                    "workers": [{k: v for k, v in w.items() if k != "public_raw"} for w in source],
                    "comparisons": []}
        for row in helper.expected_comparisons(keyed):
            combined["comparisons"].append({**{k: row[k] for k in ("left", "right", "point")},
                                           "shared_physical_born_repair_numerator_equal": row["shared"],
                                           "host_checked": False, "native_basis_identity_claimed": False,
                                           "coefficients": [helper.delta(a, b) for a, b in zip(row["left_values"], row["right_values"])]})
        self.assertEqual(helper.summarize(combined)["scalar_comparisons_reported"], 384)
        bad = copy.deepcopy(combined); bad["comparisons"][0]["coefficients"][1]["symmetric"] = 1
        with self.assertRaises(ValueError): helper.summarize(bad)
        bad = copy.deepcopy(combined); bad["comparisons"][0] = None
        with self.assertRaises(ValueError): helper.summarize(bad)

    def test_closed_json_requires_hash_finite_constants_and_size_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "closed.json"
            for raw in (b'{"x":NaN}', b'{"x":Infinity}'):
                path.write_bytes(raw)
                with self.assertRaises(ValueError): helper.read_closed(path, hashlib.sha256(raw).hexdigest())
            path.write_bytes(b"[]")
            with self.assertRaises(ValueError): helper.read_closed(path, "0" * 64)
            with mock.patch.object(helper, "MAX_INPUT_BYTES", 1), self.assertRaises(ValueError):
                helper.read_closed(path, hashlib.sha256(b"[]").hexdigest())


if __name__ == "__main__":
    unittest.main()
