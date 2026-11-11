"""Pure synthetic checks; importing or running this file launches no worker."""

import copy
import importlib.util
import math
from pathlib import Path
import statistics
import unittest


SPEC = importlib.util.spec_from_file_location("current_source_timing_analyze", Path(__file__).with_name("analyze.py"))
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)


def fixture():
    initialization, calibration = {}, {}
    for backend in ANALYZE.BACKENDS:
        key = str(backend)
        initialization[key] = {"points": 100, "ndim": 1, "nso": 1, "nborn": 1,
                               "cold_seconds": backend + 0.125}
        calibration[key] = {"target_seconds": 2.0, "loops": backend,
                            "numerical_regression_pass": True,
                            "ordinary_target_passed": True}
    cycles = []
    for entry in ANALYZE.schedule():
        cycle = entry["cycle"]
        observations = {}
        for backend in ANALYZE.BACKENDS:
            key = str(backend)
            loops = calibration[key]["loops"]
            reference = 0.001 if cycle % 2 else 0.01
            ratio = (1 if cycle <= 5 else 9) if backend == 1 else backend / 8
            # Include genuine TIRust QP use and the CutTools QP alias whose U=9
            # must never be interpreted as TIRust.
            qp_code = {1: 319, 9: 323}.get(backend)
            rows, final = [], {}
            for point in range(1, 101):
                code = qp_code if qp_code is not None and point <= 12 else 210 + backend
                final[str(code)] = final.get(str(code), 0) + 1
                for order in range(2):
                    rows.append({"point": point, "order": order, "status": code,
                                 "coefficients": [1.0, 2.0, 3.0, 4.0], "accuracy": 1e-10,
                                 "actual_backend": backend,
                                 "precision": "QP" if code >= 300 else "DP"})
            timed = {code: count * loops for code, count in final.items()}
            usage = {f"{backend}:DP": 100 * loops}
            if qp_code is not None:
                usage = {f"{backend}:DP": 88 * loops, f"{backend}:QP": 12 * loops}
            seconds = reference * ratio * 100 * loops
            observations[key] = {"loops": loops, "observation": {
                "cold_seconds": initialization[key]["cold_seconds"], "seconds": seconds,
                "seconds_per_point": seconds / (100 * loops), "rows": rows,
                "status_counts": final, "timed_status_counts": timed,
                "backend_precision_counts": usage, "native_dp_only": qp_code is None,
                "answer_dimensions": {"allocated": 1, "loop_orders": 1, "born_orders": 1},
            }}
        cycles.append({**entry, "observations": observations})
    return cycles, initialization, calibration


def parsed(cycles, backend="9", cycle=0):
    return cycles[cycle]["observations"][backend]["observation"]


class DescriptiveSummaryTests(unittest.TestCase):
    def setUp(self):
        self.cycles, self.initialization, self.calibration = fixture()

    def summarize(self):
        return ANALYZE.summarize_case("qcd_n2_gg_ttx", self.cycles, self.initialization, self.calibration)

    def test_schedule_is_exact_and_each_position_occurs_twice(self):
        self.assertEqual(ANALYZE.schedule()[0], {"cycle": 1, "order": [1, 6, 7, 8, 9]})
        self.assertEqual(ANALYZE.schedule()[5], {"cycle": 6, "order": [9, 8, 7, 6, 1]})
        for backend in ANALYZE.BACKENDS:
            self.assertEqual(sorted(entry["order"].index(backend) for entry in ANALYZE.schedule()),
                             [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    def test_summary_is_descriptive_and_does_not_mutate_inputs(self):
        before = copy.deepcopy((self.cycles, self.initialization, self.calibration))
        result = self.summarize()
        self.assertEqual((self.cycles, self.initialization, self.calibration), before)
        self.assertFalse(result["performance_acceptance"])
        self.assertFalse(result["cpu_reservation"])
        self.assertEqual(result["clock"], "SYSTEM_CLOCK wall")
        self.assertEqual(result["scope"], "full amplitude")
        self.assertEqual(result["cycle_count"], 10)
        self.assertEqual(result["numerical_rows_scope"], "final traversal only")
        self.assertEqual(result["calibration"], self.calibration)
        self.assertNotIn("confidence_interval", result)
        self.assertNotIn("passed", result)

    def test_within_cycle_ratios_are_not_ratio_of_medians(self):
        result = self.summarize()["backends"]
        self.assertEqual([item["ratio"] for item in result["1"]["ratios_to_8"]], [1] * 5 + [9] * 5)
        self.assertEqual(result["1"]["median_ratio_to_8"], 5)
        quotient = result["1"]["median_seconds_per_point"] / result["8"]["median_seconds_per_point"]
        self.assertNotAlmostEqual(quotient, result["1"]["median_ratio_to_8"])
        self.assertEqual(result["8"]["median_ratio_to_8"], 1)

    def test_unscaled_mad_and_full_evaluation_normalization(self):
        lane = self.summarize()["backends"]["9"]
        values = [sample["seconds_per_point"] for sample in lane["samples"]]
        middle = statistics.median(values)
        self.assertEqual(lane["median_seconds_per_point"], middle)
        self.assertEqual(lane["mad_seconds_per_point"], statistics.median(abs(x - middle) for x in values))
        self.assertEqual(lane["mad_scale"], "unscaled")
        self.assertEqual(lane["min_seconds_per_point"], min(values))
        self.assertEqual(lane["max_seconds_per_point"], max(values))
        sample = lane["samples"][0]
        self.assertEqual(sample["evaluations"], 900)
        self.assertEqual(sample["points"], 100)
        self.assertEqual(sample["seconds_per_point"], sample["seconds"] / 900)

    def test_timed_precision_usage_is_preserved_and_aliased_codes_are_distinct(self):
        lanes = self.summarize()["backends"]
        tirust = lanes["9"]
        self.assertFalse(tirust["native_dp_only"])
        self.assertEqual(tirust["aggregate_backend_precision_counts"], {"9:DP": 7920, "9:QP": 1080})
        self.assertEqual(tirust["samples"][0]["status_counts"], {"323": 108, "219": 792})
        self.assertEqual(tirust["samples"][0]["final_traversal_status_counts"], {"323": 12, "219": 88})
        self.assertEqual(lanes["1"]["aggregate_backend_precision_counts"], {"1:DP": 880, "1:QP": 120})
        self.assertTrue(lanes["7"]["native_dp_only"])

    def test_all_and_only_admitted_cases(self):
        self.assertEqual(len(ANALYZE.CASE_IDS), 4)
        self.assertIn("qcd_n4_uux_epem_mupmum", ANALYZE.CASE_IDS)
        for case_id in ANALYZE.CASE_IDS:
            self.assertEqual(ANALYZE.summarize_case(case_id, self.cycles, self.initialization,
                                                  self.calibration)["case_id"], case_id)
        with self.assertRaises(ValueError):
            ANALYZE.summarize_case("unregistered-case", self.cycles, self.initialization, self.calibration)

    def test_missing_duplicate_or_reordered_cycles_are_rejected(self):
        for change in (lambda c: c.pop(), lambda c: c.append(c[-1]),
                       lambda c: c.__setitem__(1, c[0]),
                       lambda c: c[0]["order"].reverse(),
                       lambda c: c[0].__setitem__("cycle", True),
                       lambda c: c[0]["order"].__setitem__(0, True)):
            with self.subTest(change=change):
                self.cycles = fixture()[0]
                change(self.cycles)
                with self.assertRaises(ValueError):
                    self.summarize()

    def test_missing_extra_or_wrong_type_backend_keys_are_rejected(self):
        for mapping in (self.initialization, self.calibration, self.cycles[0]["observations"]):
            value = mapping.pop("7")
            with self.assertRaises(ValueError):
                self.summarize()
            mapping[7] = value
            with self.assertRaises(ValueError):
                self.summarize()
            del mapping[7]
            mapping["7"] = value

    def test_bad_times_counts_and_loops_fail(self):
        for name, bad in (("seconds", 0), ("seconds", -1), ("seconds", math.nan),
                          ("seconds", math.inf), ("seconds", True),
                          ("seconds_per_point", 123), ("cold_seconds", 4)):
            with self.subTest(name=name, bad=bad):
                self.cycles = fixture()[0]
                parsed(self.cycles)[name] = bad
                with self.assertRaises(ValueError):
                    self.summarize()
        for bad in (0, -1, True, 1.5, 10):
            self.cycles = fixture()[0]
            self.cycles[0]["observations"]["9"]["loops"] = bad
            with self.assertRaises(ValueError):
                self.summarize()

    def test_incomplete_or_fabricated_histograms_fail(self):
        for edit in (lambda o: o["timed_status_counts"].__setitem__("219", 791),
                     lambda o: o["status_counts"].__setitem__("219", 87),
                     lambda o: o.__setitem__("backend_precision_counts", {"9:DP": 900}),
                     lambda o: o.__setitem__("native_dp_only", True),
                     lambda o: o["timed_status_counts"].__setitem__("219", True),
                     lambda o: o.__setitem__("timed_status_counts", {"0219": 900})):
            self.cycles = fixture()[0]
            edit(parsed(self.cycles))
            with self.assertRaises(ValueError):
                self.summarize()

    def test_wrong_backend_failed_or_initializing_statuses_fail(self):
        for bad in (319, 339, 249, 209, 419, -1, True):
            self.cycles = fixture()[0]
            parsed(self.cycles)["timed_status_counts"] = {str(bad): 900}
            with self.assertRaises(ValueError):
                self.summarize()

    def test_rows_must_cover_all_points_orders_and_histogram(self):
        for edit in (lambda o: o["rows"].pop(),
                     lambda o: o["rows"][2].__setitem__("point", 1),
                     lambda o: o["rows"][1].__setitem__("status", 219),
                     lambda o: o["rows"][0].__setitem__("precision", "DP"),
                     lambda o: o["rows"][0]["coefficients"].__setitem__(0, math.nan),
                     lambda o: o["rows"][0].__setitem__("accuracy", -1),
                     lambda o: o["answer_dimensions"].__setitem__("allocated", 0)):
            self.cycles = fixture()[0]
            edit(parsed(self.cycles))
            with self.assertRaises(ValueError):
                self.summarize()

    def test_initialization_and_calibration_are_bound_to_batches(self):
        for field, bad in (("points", 99), ("points", True), ("ndim", 2),
                           ("nborn", -1), ("cold_seconds", math.inf)):
            self.initialization = fixture()[1]
            self.initialization["9"][field] = bad
            with self.assertRaises(ValueError):
                self.summarize()
        self.initialization = fixture()[1]
        for field, bad in (("target_seconds", 1), ("target_seconds", True),
                           ("loops", 0), ("loops", 10)):
            self.calibration = fixture()[2]
            self.calibration["9"][field] = bad
            with self.assertRaises(ValueError):
                self.summarize()

    def test_dimensions_must_match_across_backends_not_only_with_own_ready(self):
        self.initialization["7"].update(ndim=0, nso=0, nborn=0)
        for cycle in self.cycles:
            item = cycle["observations"]["7"]["observation"]
            item["answer_dimensions"] = {"allocated": 0, "loop_orders": 0, "born_orders": 0}
            item["rows"] = [row for row in item["rows"] if row["order"] == 0]
        with self.assertRaisesRegex(ValueError, "Backend READY dimensions differ"):
            self.summarize()

    def test_controller_numerical_flags_are_metadata_not_derived_from_status(self):
        self.calibration["9"]["ordinary_target_passed"] = False
        lane = self.summarize()["backends"]["9"]
        self.assertFalse(lane["ordinary_target_passed"])
        self.assertTrue(lane["numerical_regression_pass"])
        self.assertEqual(lane["numerical_check_source"], "controller-supplied calibration metadata")
        self.calibration["9"]["numerical_regression_pass"] = 1
        with self.assertRaises(ValueError):
            self.summarize()

    def test_derived_zero_or_nonfinite_ratios_are_rejected(self):
        # Nonzero input seconds can underflow on division by evaluations.
        item = parsed(self.cycles)
        item["seconds"] = 5e-324
        item["seconds_per_point"] = 0.0
        with self.assertRaises(ValueError):
            self.summarize()
        # Every ratio is finite, but Python's even-count median would overflow
        # when summing two very large middle values. Do not emit Infinity JSON.
        self.cycles = fixture()[0]
        for cycle in self.cycles:
            for backend, per_point in (("1", 1.0), ("8", 1e-308)):
                wrapper = cycle["observations"][backend]
                wrapper["observation"]["seconds"] = per_point * 100 * wrapper["loops"]
                wrapper["observation"]["seconds_per_point"] = per_point
        with self.assertRaisesRegex(ValueError, "median within-cycle ratio"):
            self.summarize()


if __name__ == "__main__":
    unittest.main()
