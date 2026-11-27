"""Strict numerical extraction from a steady warm-up, never timing evidence."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark_warmup as warmup


class TestWarmupObservation(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.ready = {"points": 3, "ndim": 3, "nso": 1, "nborn": 3, "cold_seconds": 12.5}
        self.lines = ["# call point order status Born finite single double accuracy"]
        # Bootstrap unknown data and pre-window initialization are retained,
        # but neither is eligible to supply the final traversal.
        self.lines += [f"0 1 {order} unknown NaN NaN NaN NaN NaN" for order in range(4)]
        for call in range(1, 6):
            status = 239 if call < 3 else (323 if call == 4 else 219)
            for order in range(4):
                finite = "NaN" if call < 3 else str(-call * 100 - order)
                self.lines.append(f"{call} {1 + (call-1) % 3} {order} {status} "
                                  f"{call * 1000 + order} {finite} -0.5 0.25 1e-9")
        self.write()

    def write(self, lines=None, dimensions="3 1 3\n"):
        with (self.directory / "warmup.dat").open("w", newline="") as stream:
            stream.write("\r\n".join(self.lines if lines is None else lines) + "\r\n")
        (self.directory / "dimensions.dat").write_text(dimensions)

    def parse(self, backend=9, points=3, ready=None):
        return warmup.parse_warmup(self.directory, backend, points, self.ready if ready is None else ready)

    def replace_selected(self, call, order, field, value):
        lines = list(self.lines)
        for index, line in enumerate(lines):
            fields = line.split()
            if fields[:3] == [str(call), str(1 + (call - 1) % 3), str(order)]:
                fields[field] = str(value)
                lines[index] = " ".join(fields)
        return lines

    def test_wrapped_final_window_keeps_all_born_columns_and_qp_alias(self):
        observation = self.parse()
        self.assertEqual(observation["warmup_first_call"], 3)
        self.assertEqual(observation["warmup_last_call"], 5)
        self.assertEqual(observation["warmup_call_point_order"],
                         [{"call": 3, "point": 3}, {"call": 4, "point": 1}, {"call": 5, "point": 2}])
        rows = observation["rows"]
        self.assertEqual([(row["point"], row["order"]) for row in rows],
                         [(point, order) for point in range(1, 4) for order in range(4)])
        self.assertEqual(rows[3]["coefficients"], [4003.0, -403.0, -0.5, 0.25])
        self.assertEqual((rows[0]["actual_backend"], rows[0]["precision"]), (9, "QP"))
        self.assertEqual(observation["status_counts"], {"219": 2, "323": 1})
        self.assertEqual(observation["backend_precision_counts"], {"9:DP": 2, "9:QP": 1})
        self.assertFalse(observation["native_dp_only"])
        self.assertEqual(observation["answer_dimensions"], {"allocated": 3, "loop_orders": 1, "born_orders": 3})

    def test_no_timing_or_performance_acceptance_is_inferred_from_warmup(self):
        observation = self.parse()
        self.assertEqual(observation["cold_seconds"], self.ready["cold_seconds"])
        for key in ("seconds", "seconds_per_point", "timed_status_counts"):
            self.assertIsNone(observation[key])
        self.assertEqual(observation["measurement_phase"], "steady-warmup")
        self.assertIs(observation["performance_eligible"], False)
        for name, text in observation["raw_outputs"].items():
            self.assertEqual(text.encode(), (self.directory / name).read_bytes())

    def test_status_aliases_are_decoded_with_hundreds_and_chain_membership(self):
        # U=9 means CutTools QP at H=3 and TIRust DP at H=2.
        lines = [line.replace(" 323 ", " 329 ") for line in self.lines]
        self.write(lines)
        with self.assertRaises(warmup.MeasurementError):
            self.parse(backend=9)
        with self.assertRaises(warmup.MeasurementError):
            self.parse(backend=1)
        observation = self.parse(backend=[9, 1])
        self.assertEqual(observation["backend_precision_counts"], {"9:DP": 2, "1:QP": 1})
        self.assertEqual(observation["rows"][0]["actual_backend"], 1)

    def test_invalid_final_status_is_not_replaced_by_an_earlier_steady_call(self):
        lines = [line.replace(" 239 ", " 219 ").replace(" NaN ", " 1 ") for line in self.lines]
        for order in range(4):
            lines.append(f"6 3 {order} 239 1 2 3 4 1e-9")
        self.write(lines)
        with self.assertRaisesRegex(warmup.MeasurementError, "Unexpected status"):
            self.parse()

    def test_missing_duplicate_or_out_of_range_answer_columns_are_rejected(self):
        for mutate in (lambda lines: [line for line in lines if not line.startswith("5 2 3 ")],
                       lambda lines: lines + [next(line for line in lines if line.startswith("5 2 3 "))],
                       lambda lines: [line.replace("5 2 3 ", "5 2 4 ") for line in lines]):
            with self.subTest(mutate=mutate):
                self.write(mutate(list(self.lines)))
                with self.assertRaisesRegex(warmup.MeasurementError, "answer columns"):
                    self.parse()

    def test_final_call_gap_wrong_point_or_inconsistent_status_are_rejected(self):
        examples = [
            [line.replace("5 2 ", "7 1 ") for line in self.lines],
            self.replace_selected(5, 2, 1, 1),
            self.replace_selected(5, 2, 3, 323),
        ]
        for lines in examples:
            with self.subTest(lines=lines[-4:]):
                self.write(lines)
                with self.assertRaises(warmup.MeasurementError):
                    self.parse()

    def test_nonfinite_coefficients_and_missing_stability_are_rejected_with_raw(self):
        for field, value in ((4, "NaN"), (5, "Inf"), (6, "-Inf"), (7, "NaN"),
                             (8, "NaN"), (8, "-1")):
            with self.subTest(field=field, value=value):
                self.write(self.replace_selected(5, 1, field, value))
                with self.assertRaises(warmup.MeasurementError) as caught:
                    self.parse()
                for name, text in caught.exception.raw_outputs.items():
                    self.assertEqual(text.encode(), (self.directory / name).read_bytes())

    def test_ready_and_recorded_dimensions_must_match_including_born_orders(self):
        for dimensions in ("1 1 3\n", "3 3 1\n", "3 -1 3\n", "3 1 3 0\n"):
            with self.subTest(dimensions=dimensions):
                self.write(dimensions=dimensions)
                with self.assertRaises(warmup.MeasurementError):
                    self.parse()
        self.write()
        for key, value in (("points", 4), ("ndim", 2), ("nso", 2), ("nborn", 2),
                           ("points", True), ("nso", 1.0), ("cold_seconds", 0),
                           ("cold_seconds", float("nan")), ("cold_seconds", float("inf")),
                           ("cold_seconds", -1), ("cold_seconds", True)):
            ready = dict(self.ready, **{key: value})
            with self.subTest(ready=ready), self.assertRaises(warmup.MeasurementError):
                self.parse(ready=ready)

    def test_missing_files_malformed_rows_and_nonconsecutive_blocks_are_rejected(self):
        for lines in (self.lines[:9], self.lines + ["6 3 0"],
                      self.lines + ["4 1 0 219 1 2 3 4 1e-9"],
                      self.lines + ["0 1 0 unknown NaN NaN NaN NaN NaN"]):
            with self.subTest(lines=lines[-1:]):
                self.write(lines)
                with self.assertRaises(warmup.MeasurementError):
                    self.parse()
        self.write()
        (self.directory / "warmup.dat").unlink()
        with self.assertRaises(warmup.MeasurementError) as caught:
            self.parse()
        self.assertEqual(caught.exception.raw_outputs, {"dimensions.dat": "3 1 3\n"})

    def test_invalid_backend_and_point_arguments_are_rejected(self):
        for backend in ([], [3], [True], True, None):
            with self.subTest(backend=backend), self.assertRaises(warmup.MeasurementError):
                self.parse(backend=backend)
        for count in (0, -1, True, 3.0):
            with self.subTest(count=count), self.assertRaises(warmup.MeasurementError):
                self.parse(points=count)

    def test_single_point_zero_order_native_dp_and_arbitrary_column_order(self):
        self.write(["0 1 0 999 NaN NaN NaN NaN NaN", "1 1 0 211 1 -2 -3 -4 0"],
                   dimensions="0 0 0\n")
        result = self.parse(backend=1, points=1,
                            ready={"points": 1, "ndim": 0, "nso": 0, "nborn": 0, "cold_seconds": 1})
        self.assertEqual(result["status_counts"], {"211": 1})
        self.assertTrue(result["native_dp_only"])
        reordered = copy.copy(self.lines)
        reordered[-4:] = reversed(reordered[-4:])
        self.write(reordered)
        self.assertEqual(self.parse()["rows"][-1]["order"], 3)


if __name__ == "__main__":
    unittest.main()
