"""CMS adapter contract tests using temporary evidence and mocked workers only.

No supervisor, native child, build, clone, or external command is executed.
"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("cms_native_job_tests", HERE / "native_job.py")
N = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(N)

DIMENSIONS = {"allocated": 1, "loop_orders": 1, "born_orders": 1}
IDENTITIES = [(point, order) for point in range(1, 22) for order in (0, 1)]


def numerical_line(point, order, status=219, accuracy="0.001", coefficients=None):
    coefficients = coefficients or ("1.0", "2.0", "3.0", "4.0")
    return " ".join(map(str, (point, order, status, *coefficients, accuracy)))


def warmup_line(call_index, point, order, status=219, accuracy="0.001", coefficients=None):
    return str(call_index) + " " + numerical_line(point, order, status, accuracy, coefficients)


def raw_evidence(status=219, accuracy="0.001"):
    return {
        "timing.dat": "0.25 0.5\n",
        "dimensions.dat": "1 1 1\n",
        "values.dat": "\n".join(numerical_line(p, o, status, accuracy) for p, o in IDENTITIES) + "\n",
        "timed_status.dat": f"{status} 21\n",
        "warmup.dat": "# call point order status Born finite single double accuracy\n",
    }


def write_raw(directory, raw):
    for name, value in raw.items():
        (directory / name).write_text(value)


class NoProcessTests(unittest.TestCase):
    def setUp(self):
        self.no_spawn = patch.object(
            subprocess, "Popen", side_effect=AssertionError("Native/subprocess execution is forbidden in these tests")
        )
        self.no_spawn.start()
        self.addCleanup(self.no_spawn.stop)

    def parsed(self, status=219, backend=9, accuracy="0.001"):
        with tempfile.TemporaryDirectory() as tmp:
            write_raw(Path(tmp), raw_evidence(status, accuracy))
            return N.parse_outputs(Path(tmp), [backend], 21, 1)

    def gate(self, parsed, **kwargs):
        options = {"failure": None, "warnings": [], "maps_ok": True, "unexpected": [], "expected_backend": 9}
        options.update(kwargs)
        return N.qualified(parsed, **options)


class QualificationTests(NoProcessTests):
    def test_fixed_cms_contract_and_inclusive_accuracy_limit(self):
        self.assertEqual(N.POINT_COUNT, 21)
        self.assertEqual(N.PHASE_TIMEOUT_SECONDS, 300)
        self.assertEqual(N.ACCURACY_LIMIT, 1e-3)
        parsed = self.parsed()
        self.assertEqual(parsed["answer_dimensions"], DIMENSIONS)
        self.assertEqual([(r["point"], r["order"]) for r in parsed["rows"]], IDENTITIES)
        self.assertEqual(sum(parsed["timed_status_counts"].values()), 21)
        self.assertTrue(self.gate(parsed))
        self.assertFalse(self.gate(self.parsed(accuracy="0.0010000000000000002")))

    def test_gate_independently_checks_finite_numbers_and_stability(self):
        for field, value in (("accuracy", -1.0), ("accuracy", math.nan), ("accuracy", math.inf),
                             ("coefficients", [1.0, math.inf, 3.0, 4.0]),
                             ("coefficients", [1.0, 2.0, 3.0, math.nan])):
            with self.subTest(field=field, value=value):
                parsed = self.parsed()
                parsed["rows"][0][field] = value
                self.assertFalse(self.gate(parsed))

    def test_every_total_and_split_order_row_is_required_in_order(self):
        complete = self.parsed()
        changes = [complete["rows"][:-1], complete["rows"][1:],
                   complete["rows"] + [complete["rows"][-1]],
                   complete["rows"][1:2] + complete["rows"][:1] + complete["rows"][2:],
                   [r for r in complete["rows"] if r["order"] == 1]]
        for rows in changes:
            with self.subTest(identities=[(r["point"], r["order"]) for r in rows]):
                parsed = copy.deepcopy(complete)
                parsed["rows"] = rows
                self.assertFalse(self.gate(parsed))

    def test_frozen_dimensions_and_exact_histogram_are_required(self):
        for dimensions in ({"allocated": 0, "loop_orders": 0, "born_orders": 0},
                           {"allocated": 1, "loop_orders": 0, "born_orders": 1},
                           {"allocated": 1, "loop_orders": 1, "born_orders": 0},
                           {"allocated": 2, "loop_orders": 2, "born_orders": 1}):
            parsed = self.parsed()
            parsed["answer_dimensions"] = dimensions
            self.assertFalse(self.gate(parsed))
        for histogram in ({"219": 20}, {"219": 22}, {"219": 0}, {"219": 20, "313": 1}, {"219": 21, "120": 1}):
            parsed = self.parsed()
            parsed["timed_status_counts"] = histogram
            self.assertFalse(self.gate(parsed))

    def test_valid_dp_and_qp_aliases_for_each_selected_backend(self):
        for backend, code, precision in ((1, 211, "DP"), (1, 319, "QP"),
                                         (6, 216, "DP"), (6, 318, "QP"),
                                         (7, 217, "DP"), (8, 218, "DP"),
                                         (8, 312, "QP"), (9, 219, "DP"), (9, 313, "QP")):
            for tens_offset in (0, 10):
                with self.subTest(backend=backend, code=code + tens_offset):
                    parsed = self.parsed(code + tens_offset, backend)
                    self.assertEqual(parsed["rows"][0]["actual_backend"], backend)
                    self.assertEqual(parsed["rows"][0]["precision"], precision)
                    self.assertTrue(self.gate(parsed, expected_backend=backend))

    def test_unchecked_failure_initialization_unknown_and_wrong_aliases_fail(self):
        for backend, code in ((9, 120), (9, 420), (9, 239), (9, 310), (9, 319), (1, 219)):
            with self.subTest(backend=backend, code=code):
                with self.assertRaises(N.MeasurementError):
                    self.parsed(code, backend)
                parsed = self.parsed()
                parsed["rows"][0]["status"] = code
                self.assertFalse(self.gate(parsed, expected_backend=backend))

    def test_each_evaluation_has_one_status_for_total_and_split_orders(self):
        parsed = self.parsed()
        parsed["rows"][1].update(status=313, actual_backend=9, precision="QP")
        self.assertFalse(self.gate(parsed))

    def test_warnings_identity_and_failure_gates_apply_to_complete_finite_stream(self):
        parsed = self.parsed(accuracy="0")
        for options in ({"warnings": ["##W03: omitted unstable contribution"]},
                        {"warnings": [{"line": 7, "text": "##W03", "context": ["accuracy 0"]}]},
                        {"maps_ok": False}, {"unexpected": [{"class": "ValueError"}]},
                        {"failure": {"class": "ProtocolError", "message": "missing DONE"}},
                        {"cleanup_failure": {"class": "ProtocolError", "message": "missing BYE"}}):
            with self.subTest(options=options):
                self.assertFalse(self.gate(parsed, **options))

    def test_warmup_can_never_substitute_for_final_traversal(self):
        raw = raw_evidence()
        raw.pop("values.dat")
        raw["warmup.dat"] += "\n".join(warmup_line(p, p, o) for p, o in IDENTITIES) + "\n"
        diagnostic = N.diagnostics(raw, 9)
        self.assertEqual(len(diagnostic["warmup_rows"]), 42)
        self.assertEqual(diagnostic["traversal_rows"], [])
        self.assertFalse(self.gate(None))
        self.assertTrue(all(r["state"] == "unexecuted" for r in diagnostic["phase_outcomes"]["traversal"]))


class DiagnosticTests(NoProcessTests):
    def outcome(self, diagnostic, phase, point, order):
        matches = [r for r in diagnostic["phase_outcomes"][phase] if (r["point"], r["order"]) == (point, order)]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def test_bootstrap_and_repeated_warmup_calls_keep_independent_identity(self):
        raw = {"warmup.dat": "# retained initialization calls\n" + "\n".join(
            warmup_line(call, point, order, status, accuracy)
            for call, point, status, accuracy in ((0, 1, 120, "-1"), (1, 1, 239, "-1"),
                                                (2, 2, 219, "0.0001"), (22, 1, 219, "0.0001"))
            for order in (0, 1)) + "\n"}
        diagnostic = N.diagnostics(raw, 9)
        self.assertEqual([(r["call_index"], r["point"], r["order"]) for r in diagnostic["bootstrap_rows"]],
                         [(0, 1, 0), (0, 1, 1)])
        self.assertEqual([(r["call_index"], r["point"], r["order"]) for r in diagnostic["warmup_rows"]],
                         [(1, 1, 0), (1, 1, 1), (2, 2, 0), (2, 2, 1), (22, 1, 0), (22, 1, 1)])
        self.assertTrue(all(not row["eligible"] for row in diagnostic["bootstrap_rows"]))
        self.assertTrue(all(not row["eligible"] for row in diagnostic["warmup_rows"][:2]))
        self.assertTrue(all(row["eligible"] for row in diagnostic["warmup_rows"][2:]))
        self.assertEqual(self.outcome(diagnostic, "warmup", 1, 0)["state"], "accepted")
        for phase in ("warmup", "traversal"):
            self.assertEqual([(r["point"], r["order"]) for r in diagnostic["phase_outcomes"][phase]], IDENTITIES)

    def test_early_warmup_refusal_marks_later_points_and_all_traversal_unexecuted(self):
        raw = {"warmup.dat": "\n".join(
            warmup_line(p, p, o, 420 if p == 20 else 219, "0")
            for p in range(1, 21) for o in (0, 1)) + "\n"}
        diagnostic = N.diagnostics(raw, 9)
        for order in (0, 1):
            self.assertEqual(self.outcome(diagnostic, "warmup", 19, order)["state"], "accepted")
            self.assertEqual(self.outcome(diagnostic, "warmup", 20, order)["state"], "rejected")
            self.assertEqual(self.outcome(diagnostic, "warmup", 21, order)["state"], "unexecuted")
        self.assertTrue(all(r["state"] == "unexecuted" for r in diagnostic["phase_outcomes"]["traversal"]))

    def test_partial_final_traversal_has_explicit_rejected_and_unexecuted_rows(self):
        raw = {"values.dat": numerical_line(1, 0) + "\n" + numerical_line(1, 1, 420, "0") + "\n"}
        diagnostic = N.diagnostics(raw, 9)
        self.assertEqual(self.outcome(diagnostic, "traversal", 1, 0)["state"], "accepted")
        self.assertEqual(self.outcome(diagnostic, "traversal", 1, 1)["state"], "rejected")
        self.assertEqual(self.outcome(diagnostic, "traversal", 2, 0)["state"], "unexecuted")
        self.assertTrue(all("reason" in row for row in diagnostic["traversal_rows"]))

    def test_nonfinite_diagnostics_survive_strict_json_serialization(self):
        raw = {"values.dat": numerical_line(20, 0, 420, "NaN", ("NaN", "Infinity", "-Infinity", "4")) + "\n"}
        diagnostic = N.diagnostics(raw, 9)
        self.assertEqual(len(diagnostic["traversal_rows"]), 1)
        self.assertFalse(diagnostic["traversal_rows"][0]["eligible"])
        safe = N.safe_json(diagnostic)
        encoded = json.dumps(safe, allow_nan=False)
        decoded = json.loads(encoded)
        coefficients = decoded["traversal_rows"][0]["coefficients"]
        self.assertTrue(all(isinstance(value, dict) and value for value in coefficients[:3]))
        self.assertEqual(coefficients[3], 4.0)
        self.assertIsInstance(decoded["traversal_rows"][0]["accuracy"], dict)
        self.assertEqual(json.loads(json.dumps(N.safe_json({"nested": [math.nan, math.inf, -math.inf]}), allow_nan=False))["nested"],
                         N.safe_json([math.nan, math.inf, -math.inf]))

    def test_malformed_rows_are_retained_as_parse_issues(self):
        diagnostic = N.diagnostics({"warmup.dat": "# header\nnot a numerical row\n",
                                    "values.dat": "1 0 219 1 2\n"}, 9)
        self.assertTrue(diagnostic["parse_issues"])
        self.assertEqual(len(diagnostic["phase_outcomes"]["warmup"]), 42)
        self.assertEqual(len(diagnostic["phase_outcomes"]["traversal"]), 42)


class RequestTests(NoProcessTests):
    def request(self, worker_error=None, close_error=None, parser_error=None, log="", startup=False,
                mapped=False, accuracy="0.001", raw_suffix=b""):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        directory = root / "worker"
        evidence = root / "evidence"
        directory.mkdir()
        evidence.mkdir()
        raw = raw_evidence(accuracy=accuracy)
        if startup:
            raw = {"warmup.dat": "\n".join(warmup_line(20, 20, o, 420, "0") for o in (0, 1)) + "\n",
                   "dimensions.dat": "1 1 1\n"}
        write_raw(directory, raw)
        if raw_suffix:
            values_path = directory / "values.dat"
            values_path.write_bytes(values_path.read_bytes() + raw_suffix)
        spec_path = evidence / "spec.json"
        spec_path.write_text(json.dumps({"request": "candidate-9-auto", "cwd": str(directory), "backend": 9,
                                         "argv_without_fifo": ["never-execute"], "expected_runtime": {}}))
        calls = []

        class FakeWorker:
            def __init__(self, command, cwd, log_path, env=None, timeout=None):
                self.process = types.SimpleNamespace(pid=43210) if mapped else None
                calls.append(("init", command, cwd, timeout))
                Path(log_path).write_text(log)
                if mapped:
                    self._message()
                if startup and worker_error:
                    self.close(graceful=False)
                    raise worker_error

            def _message(self):
                return ["READY", "1", "21", "1", "1", "1", "0.25"]

            def run(self, loops):
                calls.append(("run", loops))
                if worker_error:
                    raise worker_error

            def close(self, *args, **kwargs):
                calls.append(("close",))
                if close_error:
                    raise close_error

        parsed = self.parsed(accuracy=accuracy)

        def parser(cwd, backend, points, loops):
            calls.append(("parse", cwd, backend, points, loops))
            if parser_error:
                raise parser_error
            return parsed

        read_text = Path.read_text

        def fixture_read_text(path, *args, **kwargs):
            if str(path).startswith("/proc/"):
                self.assertEqual(str(path), "/proc/43210/maps")
                return "mocked process map; no live process inspected\n"
            return read_text(path, *args, **kwargs)

        def classify(raw_maps, expected):
            calls.append(("maps", raw_maps, expected))
            return {"elf_files": {}, "data_files": {}, "exact_expected": True}

        with patch.object(Path, "read_text", fixture_read_text):
            result = N.run_request(spec_path, worker_class=FakeWorker, parser=parser, maps_classifier=classify)
        retained = json.loads((evidence / "result.json").read_text(), parse_constant=lambda value: self.fail(value))
        self.assertEqual(result["numerically_qualified"], retained["numerically_qualified"])
        return result, retained, calls, evidence

    def test_mocked_request_uses_full_stream_and_cms_phase_deadline(self):
        result, retained, calls, evidence = self.request()
        self.assertEqual(calls[0][3], 300)
        self.assertIn(("run", 1), calls)
        self.assertEqual(next(row for row in calls if row[0] == "parse")[2:], ([9], 21, 1))
        self.assertIn(("close",), calls)
        self.assertFalse(result["fatal_controller_failure"])
        self.assertFalse(result["numerically_qualified"])
        self.assertFalse(result["complete_runtime_map"])
        self.assertEqual(set(retained["raw_outputs_sha256"]), set(raw_evidence()))
        self.assertTrue((evidence / "warmup.dat").exists())

    def test_fully_mocked_complete_request_can_qualify(self):
        result, retained, calls, _ = self.request(mapped=True)
        self.assertTrue(result["numerically_qualified"])
        self.assertFalse(result["fatal_controller_failure"])
        self.assertTrue(result["complete_runtime_map"])
        self.assertEqual(len(retained["actual_map_captures"]), 1)
        self.assertTrue(any(row[0] == "maps" for row in calls))

    def test_shutdown_failure_denies_qualification_and_is_fatal_with_result_retained(self):
        result, retained, calls, _ = self.request(close_error=N.ProtocolError("Missing BYE"), mapped=True)
        self.assertFalse(result["numerically_qualified"])
        self.assertTrue(result["fatal_controller_failure"])
        self.assertEqual(result["cleanup_failure"]["class"], "ProtocolError")
        self.assertIn("Missing BYE", retained["cleanup_failure"]["message"])
        self.assertIn(("close",), calls)

    def test_retained_protocol_and_measurement_errors_are_nonfatal(self):
        for error in (N.ProtocolError("bounded protocol timeout"), N.MeasurementError("rejected row", {})):
            with self.subTest(error=type(error).__name__):
                result, retained, calls, _ = self.request(worker_error=error)
                self.assertFalse(result["numerically_qualified"])
                self.assertFalse(result["fatal_controller_failure"])
                self.assertEqual(retained["failure"]["class"], type(error).__name__)
                self.assertIn(("close",), calls)

    def test_measurement_parse_rejection_is_nonfatal_but_unexpected_parser_error_is_fatal(self):
        for error, fatal in ((N.MeasurementError("incomplete final rows", {}), False),
                             (RuntimeError("unexpected controller invariant"), True)):
            with self.subTest(error=type(error).__name__):
                result, retained, _, _ = self.request(parser_error=error)
                self.assertEqual(result["fatal_controller_failure"], fatal)
                self.assertFalse(retained["numerically_qualified"])

    def test_unexpected_worker_controller_exception_is_fatal(self):
        result, retained, calls, _ = self.request(worker_error=RuntimeError("controller invariant"))
        self.assertTrue(result["fatal_controller_failure"])
        self.assertFalse(retained["numerically_qualified"])
        self.assertIn(("close",), calls)

    def test_constructor_cleanup_failure_is_fatal_and_retained(self):
        result, retained, _, _ = self.request(worker_error=N.MeasurementError("warmup refused", {}),
                                               close_error=N.ProtocolError("startup cleanup failed"), startup=True)
        self.assertTrue(result["fatal_controller_failure"])
        self.assertFalse(result["numerically_qualified"])
        self.assertIn("startup cleanup failed", retained["cleanup_failure"]["message"])

    def test_raw_output_bytes_and_hash_are_preserved_before_diagnostic_decoding(self):
        suffix = b"\xff\xfe\n"
        result, retained, _, evidence = self.request(raw_suffix=suffix)
        expected = raw_evidence()["values.dat"].encode() + suffix
        self.assertEqual((evidence / "values.dat").read_bytes(), expected)
        self.assertEqual(retained["raw_outputs_sha256"]["values.dat"], hashlib.sha256(expected).hexdigest())
        self.assertFalse(result["numerically_qualified"])
        self.assertTrue(retained["diagnostics"]["parse_issues"])

    def test_startup_refusal_preserves_warmup_and_missing_final_coverage(self):
        result, retained, calls, _ = self.request(
            worker_error=N.MeasurementError("warmup status420 point20", {}), startup=True
        )
        self.assertFalse(result["fatal_controller_failure"])
        self.assertFalse(result["numerically_qualified"])
        self.assertFalse(any(row[0] == "run" for row in calls))
        diagnostic = retained["diagnostics"]
        self.assertEqual(len(diagnostic["warmup_rows"]), 2)
        self.assertEqual(diagnostic["traversal_rows"], [])
        self.assertTrue(all(row["state"] == "unexecuted" for row in diagnostic["phase_outcomes"]["traversal"]))

    def test_w03_retains_line_and_context_even_with_finite_zero_accuracy(self):
        result, retained, _, _ = self.request(log="before diagnostic\n##W03: unstable omitted term\nafter diagnostic\n",
                                              mapped=True, accuracy="0")
        self.assertFalse(result["numerically_qualified"])
        self.assertEqual(len(retained["incomplete_warnings"]), 1)
        warning = retained["incomplete_warnings"][0]
        self.assertIsInstance(warning, dict)
        self.assertIn("##W03", json.dumps(warning))
        self.assertIn("before diagnostic", json.dumps(warning))
        self.assertIn("after diagnostic", json.dumps(warning))
        self.assertTrue(any("line" in key for key in warning))


if __name__ == "__main__":
    unittest.main()
