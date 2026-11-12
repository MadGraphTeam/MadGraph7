"""Pure protocol counterexamples; no benchmark, reducer or SDK imports."""
import copy
import importlib.util
from pathlib import Path
import unittest
import ast
import json
import sys
import tempfile
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("private_route_statistics", Path(__file__).resolve().parents[3] / "tools/tirust_benchmark_worker.py")
routes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routes)
sys.modules["tirust_benchmark_worker"] = routes
ROOT = Path(__file__).resolve().parents[3]
b_spec = importlib.util.spec_from_file_location("private_benchmark", ROOT / "tools/tirust_benchmark.py")
bench = importlib.util.module_from_spec(b_spec)
b_spec.loader.exec_module(bench)


def text(delta=None, before=None, statuses=(0, 0), sequence=2):
    delta = [0] * 20 if delta is None else list(delta)
    before = [0] * 20 if before is None else list(before)
    after = [a + b for a, b in zip(before, delta)]
    return " ".join(map(str, [1, sequence, 3, 100, *statuses, *before, *after, *delta]))


def parsed(delta=None, **kwargs):
    return routes.parse_route_statistics(text(delta, **kwargs), 100, 3, sequence=2)


def usage(delta=None, counts=None, selected=9, **kwargs):
    obs = {"backend_precision_counts": {"9:DP": 300} if counts is None else counts,
           "native_dp_only": True, "route_statistics": parsed(delta)}
    return routes.precision_usage(obs, selected, **kwargs)


class RouteEvidenceTests(unittest.TestCase):
    def test_exact_header_blocks_and_delta(self):
        d = list(range(20)); b = [10] * 20
        r = parsed(d, before=b)
        self.assertEqual(r["delta"], d)
        self.assertEqual(r["primary"]["unreported"], 6)
        self.assertEqual(r["check"]["unreported"], 16)
        self.assertTrue(r["available"])

    def test_every_snapshot_counter_is_monotonic_and_exact(self):
        for k in range(20):
            words = text([1] * 20, [2] * 20).split()
            words[26 + k] = "1"
            with self.subTest(counter=k), self.assertRaises(ValueError):
                routes.parse_route_statistics(" ".join(words), 100, 3)
            words = text([1] * 20).split(); words[46 + k] = "0"
            with self.subTest(delta=k), self.assertRaises(ValueError):
                routes.parse_route_statistics(" ".join(words), 100, 3)

    def test_malformed_ascii_integer_count_and_ranges(self):
        invalid = [text() + " 0", " ".join(text().split()[:-1]), text().replace(" 100 ", " 1e2 ", 1),
                   text().replace(" 100 ", " 100.0 ", 1), text().replace(" 100 ", " １００ ", 1),
                   text().replace(" 100 ", " True ", 1), text().replace(" 100 ", " -0 ", 1),
                   text().replace(" 100 ", " 9223372036854775808 ", 1)]
        for value in invalid:
            with self.subTest(text=value[:40]), self.assertRaises(ValueError):
                routes.parse_route_statistics(value, 100, 3)

    def test_boolean_arguments_and_count_overflow(self):
        for p, n, seq in [(True, 3, 2), (100, True, 2), (100, 3, True),
                          (0, 3, 2), (100, -1, 2), (routes.MAX_I64, 2, 2)]:
            with self.subTest(args=(p, n, seq)), self.assertRaises(ValueError):
                routes.parse_route_statistics(text(), p, n, seq)

    def test_stale_sequence_wrong_counts_and_schema(self):
        for field, replacement in [(0, "2"), (1, "1"), (2, "2"), (3, "99")]:
            words = text().split(); words[field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                routes.parse_route_statistics(" ".join(words), 100, 3, 2)

    def test_nonzero_getter_status_is_unknown(self):
        for status in [(1, 0), (0, 1), (999, 2)]:
            r = parsed([1] * 20, statuses=status)
            u = routes.precision_usage({"backend_precision_counts": {"9:DP": 300}, "route_statistics": r}, 9)
            self.assertFalse(r["available"])
            self.assertIsNone(u["native_dp_only"])
            self.assertIsNone(u["bridge_native_dp_only"])
            self.assertIsNone(u["internal_higher_precision_attempted"])

    def test_getter_failure_zeroing_positive_prior_counts_is_valid_unknown(self):
        before = [100] * 20; after = [0] * 20; delta = [-100] * 20
        words = [1, 2, 3, 100, 0, 1, *before, *after, *delta]
        r = routes.parse_route_statistics(" ".join(map(str, words)), 100, 3, 2)
        self.assertFalse(r["available"])
        self.assertEqual(r["delta"], delta)
        u = routes.precision_usage({"backend_precision_counts": {"9:DP": 300}, "route_statistics": r}, 9)
        self.assertIsNone(u["native_dp_only"])
        self.assertIsNone(u["bridge_native_dp_only"])
        words[5] = 0
        with self.assertRaises(ValueError):
            routes.parse_route_statistics(" ".join(map(str, words)), 100, 3, 2)
        words[5] = 1; words[-1] = -99
        with self.assertRaises(ValueError):
            routes.parse_route_statistics(" ".join(map(str, words)), 100, 3, 2)

    def test_legacy_true_flag_is_not_arithmetic_evidence(self):
        obs = {"backend_precision_counts": {"9:DP": 300}, "native_dp_only": True}
        before = copy.deepcopy(obs)
        u = routes.precision_usage(obs, 9)
        self.assertEqual(obs, before)
        self.assertIsNone(u["native_dp_only"])
        self.assertIsNone(u["bridge_native_dp_only"])

    def test_no_actual_bridge_computations_is_unknown(self):
        u = usage()
        self.assertEqual(u["route_evidence_status"], "no-bridge-computations")
        self.assertIsNone(u["bridge_native_dp_only"])

    def test_complete_binary64_bridge_interval_does_not_prove_full_amplitude(self):
        d = [0] * 20; d[0], d[11] = 51, 47
        u = usage(d)
        self.assertTrue(u["bridge_native_dp_only"])
        self.assertIsNone(u["native_dp_only"])
        self.assertFalse(u["internal_higher_precision_attempted"])

    def test_primary_or_check_only_is_unknown(self):
        for index in (0, 1, 10, 11):
            d = [0] * 20; d[index] = 8
            self.assertIsNone(usage(d)["bridge_native_dp_only"])

    def test_every_success_and_attempt_tier_in_both_passes_rules_out_native_dp(self):
        for offset in (0, 10):
            for index, bits in [(2, 106), (3, 256), (4, 512), (7, 106), (8, 256), (9, 512)]:
                d = [0] * 20; d[offset + index] = 1
                u = usage(d)
                with self.subTest(offset=offset, index=index):
                    self.assertFalse(u["native_dp_only"])
                    self.assertFalse(u["bridge_native_dp_only"])
                    self.assertTrue(u["internal_higher_precision_attempted"])
                    self.assertEqual(u["highest_observed_internal_bits"], bits)
                    self.assertEqual(u["precision_classification"], "internal higher-precision attempted")

    def test_failed_attempt_is_not_success_and_counts_not_double_counted(self):
        d = [0] * 20; d[8] = 7
        u = usage(d)
        self.assertEqual(u["higher_precision_attempt_counts"]["primary"]["256"], 7)
        self.assertEqual(u["higher_precision_success_counts"]["primary"]["256"], 0)
        self.assertFalse(u["internal_higher_precision_success_observed"])
        self.assertNotIn("rescued_calls", u)

    def test_unreported_censors_zero_attempts_but_positive_tier_still_rules_out_dp(self):
        for index in (6, 16):
            d = [0] * 20; d[0] = d[10] = 9; d[index] = 1
            u = usage(d)
            self.assertEqual(u["attempt_count_scope"], "lower bounds")
            self.assertIsNone(u["native_dp_only"])
            self.assertIsNone(u["bridge_native_dp_only"])
            self.assertIsNone(u["internal_higher_precision_attempted"])
            d[9] = 1
            self.assertFalse(usage(d)["native_dp_only"])

    def test_preexisting_unreported_is_history_not_current_interval_censoring(self):
        d = [0] * 20; d[0] = d[10] = 5; before = [0] * 20; before[6] = 1
        r = parsed(d, before=before)
        u = routes.precision_usage({"backend_precision_counts": {"9:DP": 300}, "route_statistics": r}, 9)
        self.assertTrue(u["bridge_native_dp_only"])
        self.assertIsNone(u["native_dp_only"])
        self.assertEqual(u["attempt_count_scope"], "reported counts")

    def test_refused_route_is_unknown_without_higher_precision_observation(self):
        d = [0] * 20; d[0] = d[10] = 5; d[5] = 1
        self.assertIsNone(usage(d)["bridge_native_dp_only"])

    def test_qp_transport_and_numerator_card_are_separate(self):
        for card in (None, False, True):
            u = usage(counts={"9:QP": 300}, use_qp_integrand_for_tirust=card)
            self.assertTrue(u["tirust_qp_transport_observed"])
            self.assertIs(u["tirust_qp_numerator_enabled"], card)
            self.assertFalse(u["native_dp_only"])
        with self.assertRaises(ValueError):
            usage(use_qp_integrand_for_tirust=1)

    def test_fallback_backends_rule_out_native_dp(self):
        for counts, selected in [({"6:DP": 300}, [9, 6]), ({"9:DP": 300}, [6, 9]),
                                 ({"9:DP": 10, "6:DP": 290}, None)]:
            u = routes.precision_usage({"backend_precision_counts": counts}, selected)
            self.assertTrue(u["fallback_backend_observed"])
            self.assertFalse(u["native_dp_only"])

    def test_parsed_map_tampering_cannot_override_raw_evidence(self):
        r = parsed(); r["available"] = True; r["primary"]["direct"] = 100
        u = routes.precision_usage({"backend_precision_counts": {"9:DP": 300}, "route_statistics": r}, 9)
        self.assertIsNone(u["bridge_native_dp_only"])
        r["before"][0] = True
        with self.assertRaises(ValueError):
            routes.precision_usage({"route_statistics": r})

    def test_backend_count_and_selection_type_guards(self):
        for counts in [{"9:DP": True}, {"7:QP": 1}, {"9:DP": 0}, {"9:DD": 1}, {"9:DP": routes.MAX_I64 + 1}]:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                routes.precision_usage({"backend_precision_counts": counts}, 9)
        for selected in [True, [], [True], [3], "9"]:
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                routes.precision_usage({}, selected)

    def test_three_valued_aggregation(self):
        for flags, expected in [([], None), ([True], True), ([True, True], True),
                                ([None], None), ([True, None], None), ([False, None], False),
                                ([False, True], False)]:
            self.assertIs(routes.aggregate_native_dp_only(flags), expected)
        with self.assertRaises(ValueError):
            routes.aggregate_native_dp_only([1])

    def test_aggregate_empty_zero_and_missing_intervals_remain_unknown(self):
        a = routes.aggregate_precision_usage([])
        self.assertIsNone(a["route_counter_delta"])
        self.assertIsNone(a["native_dp_only"])
        self.assertEqual(a["route_interval_coverage"], "unknown")
        zero = {"backend_precision_counts": {"9:DP": 300}, "route_statistics": parsed()}
        a = routes.aggregate_precision_usage([zero], 9)
        self.assertEqual(a["route_counter_delta"], [0] * 20)
        self.assertIsNone(a["bridge_native_dp_only"])
        a = routes.aggregate_precision_usage([zero, {"backend_precision_counts": {"9:DP": 300}}], 9)
        self.assertEqual(a["route_interval_coverage"], "partial")
        self.assertEqual(a["attempt_count_scope"], "lower bounds")
        self.assertIsNone(a["native_dp_only"])

    def test_aggregate_known_attempts_dominate_unknown_and_do_not_double_count(self):
        d = [0] * 20; d[3], d[8], d[19] = 2, 3, 4
        o = {"backend_precision_counts": {"9:DP": 300}, "route_statistics": parsed(d)}
        missing = {"backend_precision_counts": {"9:DP": 300}}
        a = routes.aggregate_precision_usage([o, o, missing], 9)
        self.assertFalse(a["native_dp_only"])
        self.assertFalse(a["bridge_native_dp_only"])
        self.assertTrue(a["internal_higher_precision_attempted"])
        self.assertEqual(a["highest_observed_internal_bits"], 512)
        self.assertEqual(a["route_counter_delta"], [2 * v for v in d])
        self.assertEqual(a["higher_precision_success_counts"]["primary"]["256"], 4)
        self.assertEqual(a["higher_precision_attempt_counts"]["primary"]["256"], 6)
        self.assertEqual(a["higher_precision_attempt_counts"]["check"]["512"], 8)
        self.assertEqual(a["accepted_backend_precision_counts"], {"9:DP": 900})
        self.assertEqual(a["attempt_count_scope"], "lower bounds")

    def test_aggregate_complete_bridge_dp_cannot_certify_fortran_cache_bypasses(self):
        d = [0] * 20; d[0] = d[10] = 3
        o = {"backend_precision_counts": {"9:DP": 300}, "route_statistics": parsed(d)}
        a = routes.aggregate_precision_usage([o, o], 9)
        self.assertTrue(a["bridge_native_dp_only"])
        self.assertIsNone(a["native_dp_only"])
        self.assertEqual(a["route_interval_coverage"], "complete")
        self.assertEqual(a["attempt_count_scope"], "reported counts")

    def test_aggregate_unavailable_and_qp_numerator_card_are_separate(self):
        o = {"backend_precision_counts": {"9:QP": 300}, "route_statistics": parsed(statuses=(0, 1))}
        for card in (None, False, True):
            a = routes.aggregate_precision_usage([o], 9, card)
            self.assertFalse(a["native_dp_only"])
            self.assertIs(a["tirust_qp_numerator_enabled"], card)
            self.assertIsNone(a["route_counter_delta"])
            self.assertEqual(a["route_interval_coverage"], "partial")
            self.assertIn("no whole-amplitude precision claim", a["whole_amplitude_qp_scope"])



class DriverAndRetentionTests(unittest.TestCase):
    def parameters(self, enabled=False):
        return {"prefix": "ML5_0_", "nexternal": 3, "nborn": 0,
                "tirust_dp_route_statistics": enabled}

    def fixture(self, directory):
        (directory / "dimensions.dat").write_text("0 0 0\n")
        (directory / "timing.dat").write_text("0.1 0.2\n")
        (directory / "values.dat").write_text("1 0 219 1 2 -3 -4 1e-10\n")
        (directory / "timed_status.dat").write_text("219 3\n")

    def route(self, delta=None, statuses=(0, 0), sequence=1, before=None):
        before = [0] * 20 if before is None else list(before)
        delta = [0] * 20 if delta is None else list(delta)
        after = [a + b for a, b in zip(before, delta)]
        return " ".join(map(str, [1, sequence, 3, 1, *statuses, *before, *after, *delta])) + "\n"

    def test_capability_exact_true_only_and_baseline_disabled(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder); (prefix / "lib").mkdir()
            cfg = prefix / "lib/tirust-config.json"
            self.assertFalse(bench.tirust_dp_route_statistics_enabled("candidate", prefix))
            for value in (None, False, 0, 1, "True", "true"):
                cfg.write_text(json.dumps({"dp_route_statistics": value}))
                self.assertFalse(bench.tirust_dp_route_statistics_enabled("candidate", prefix))
            cfg.write_text('{"dp_route_statistics": true}')
            self.assertTrue(bench.tirust_dp_route_statistics_enabled("candidate", prefix))
            self.assertFalse(bench.tirust_dp_route_statistics_enabled("baseline", prefix))
            cfg.write_text('[]')
            with self.assertRaises(ValueError):
                bench.tirust_dp_route_statistics_enabled("candidate", prefix)

    def test_legacy_driver_has_no_new_symbol_or_module_dependency(self):
        template = (ROOT / "tools/tirust_benchmark_worker.f").read_text()
        old = bench.render_worker(template, self.parameters(False))
        self.assertNotIn("@TIRUST_ROUTE_", old)
        self.assertNotIn("GET_TIRUST_DP_ROUTE_STATISTICS", old)
        self.assertNotIn("USE MTIRUST", old)
        historical = "      CALL @PREFIX@GET_ANSWER_DIMENSION(@NEXTERNAL@)\nC @NBORN@\n"
        self.assertEqual(bench.render_worker(historical, self.parameters(True)),
            "      CALL ML5_0_GET_ANSWER_DIMENSION(3)\nC 0\n")

    def test_three_token_legacy_renderer_leaves_only_harmless_comment_anchors(self):
        template = (ROOT / "tools/tirust_benchmark_worker.f").read_text()
        legacy = template.replace("@PREFIX@", "OLD_").replace("@NEXTERNAL@", "3").replace("@NBORN@", "0")
        anchors = [line for line in legacy.splitlines() if "@TIRUST_ROUTE_" in line]
        self.assertEqual(len(anchors), 4)
        self.assertTrue(all(line.startswith("C@TIRUST_ROUTE_") for line in anchors))
        executable = "\n".join(line for line in legacy.splitlines()
                                 if line and line[0] not in "Cc*!" )
        self.assertNotIn("@TIRUST_ROUTE_", executable)
        with self.assertRaisesRegex(ValueError, "whole fixed-form comment"):
            bench.render_worker("      X='C@TIRUST_ROUTE_BEFORE@'", self.parameters(True))

    def test_both_getters_and_all_route_io_are_outside_timed_loop(self):
        text = bench.render_worker((ROOT / "tools/tirust_benchmark_worker.f").read_text(), self.parameters(True))
        run = text.index("      TIMED_CODES=0")
        start = text.index("      CALL SYSTEM_CLOCK(T0)", run)
        end = text.index("      CALL SYSTEM_CLOCK(T1)", start)
        getter = "CALL ML5_0_GET_TIRUST_DP_ROUTE_STATISTICS"
        self.assertEqual(text.count(getter), 2)
        self.assertEqual(text[:start].count(getter), 1)
        self.assertEqual(text[end:].count(getter), 1)
        self.assertNotIn("TIRUST_ROUTE", text[start:end])
        self.assertGreater(text.index("FILE='tirust_routes.dat'"), end)
        self.assertNotIn("@TIRUST_ROUTE_", text)
        self.assertLessEqual(max(map(len, text.splitlines())), 132)

    def test_partial_and_duplicate_hook_templates_refuse(self):
        with self.assertRaises(ValueError):
            bench.render_worker("C@TIRUST_ROUTE_BEFORE@", self.parameters(True))
        template = (ROOT / "tools/tirust_benchmark_worker.f").read_text()
        with self.assertRaises(ValueError):
            bench.render_worker(template + "\nC@TIRUST_ROUTE_BEFORE@", self.parameters(True))
        with self.assertRaises(ValueError):
            bench.render_worker(template, {**self.parameters(), "tirust_dp_route_statistics": 1})

    def test_legacy_observation_dictionary_is_unchanged_but_current_usage_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder);self.fixture(directory)
            result = routes.parse_outputs(directory, 9, 1, 3)
            self.assertEqual(set(result), {"cold_seconds", "seconds", "seconds_per_point", "rows",
                "status_counts", "timed_status_counts", "backend_precision_counts", "native_dp_only", "answer_dimensions"})
            self.assertTrue(result["native_dp_only"])  # retained dispatch-scoped historical alias
            current = routes.precision_usage(result, selected=9)
            self.assertIsNone(current["native_dp_only"])
            self.assertEqual(current["route_evidence_status"], "absent")
            self.assertEqual(result["backend_precision_counts"], {"9:DP": 3})

    def test_219_with_attempted_tier_is_not_native_and_transport_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder);self.fixture(directory)
            delta = [0] * 20;delta[2] = delta[7] = 1
            (directory / routes.ROUTE_STATISTICS_OUTPUT).write_text(self.route(delta))
            result = routes.parse_outputs(directory, 9, 1, 3)
            self.assertFalse(result["native_dp_only"])
            self.assertTrue(result["precision_usage"]["internal_higher_precision_attempted"])
            self.assertEqual(result["backend_precision_counts"], {"9:DP": 3})
            self.assertNotIn("9:QP", result["backend_precision_counts"])

    def test_unavailable_route_snapshot_keeps_numerics_but_precision_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder);self.fixture(directory)
            before = [5] + [0] * 19; delta = [-5] + [0] * 19
            (directory / routes.ROUTE_STATISTICS_OUTPUT).write_text(self.route(delta, (0, 2), before=before))
            result = routes.parse_outputs(directory, 9, 1, 3)
            self.assertIsNone(result["native_dp_only"])
            self.assertFalse(result["route_statistics"]["available"])
            self.assertEqual(result["seconds"], .2)

    def test_rejected_sidecar_retains_exact_raw_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder);self.fixture(directory)
            raw = self.route().replace("1 ", "9 ", 1)
            (directory / routes.ROUTE_STATISTICS_OUTPUT).write_text(raw)
            with self.assertRaises(routes.MeasurementError) as caught:
                routes.parse_outputs(directory, 9, 1, 3)
            self.assertEqual(caught.exception.raw_outputs[routes.ROUTE_STATISTICS_OUTPUT], raw)

    def test_worker_clears_stale_sidecar_and_archives_new_one_without_process(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "run";directory.mkdir()
            archive = Path(folder) / "archive";archive.mkdir()
            sidecar = directory / routes.ROUTE_STATISTICS_OUTPUT;sidecar.write_text("stale")
            worker = object.__new__(routes.Worker)
            worker.directory = directory;worker.raw_directory = archive;worker.sequence = 0
            worker.closed = False;worker.points = 1;worker.log = Path(folder) / "log"
            worker.close = lambda **kwargs: None
            def send(line):
                self.assertFalse(sidecar.exists())
                self.assertEqual(line, "RUN 1 3")
                self.fixture(directory);sidecar.write_text(self.route())
            worker._send = send;worker._message = lambda: ["DONE", "1", "3"]
            worker.run(3)
            self.assertEqual((worker.last_raw_directory / routes.ROUTE_STATISTICS_OUTPUT).read_text(), self.route())
            target = Path(folder) / "retained"
            bench.retain_observation(directory, target)
            self.assertEqual((target / routes.ROUTE_STATISTICS_OUTPUT).read_text(), self.route())

    def test_fresh_tirust_cohort_refuses_before_any_preparation_or_execution(self):
        source = ast.parse((ROOT / "tools/tirust_benchmark_cohorts.py").read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<cohort-guard>", "exec"), namespace)
        with self.assertRaisesRegex(ValueError, "per-call route adapter"):
            namespace["run"](SimpleNamespace(backend=9))

    def test_current_aggregation_ignores_legacy_native_alias_and_checks_qp_card(self):
        legacy = {"backend_precision_counts": {"9:DP": 1}, "native_dp_only": True}
        self.assertIsNone(bench.aggregate_precision_usage([legacy], [9])["native_dp_only"])
        qp = {"backend_precision_counts": {"9:QP": 1}, "native_dp_only": False}
        for value in (True, False, None):
            settings = {} if value is None else {"useqpintegrandfortirust": value}
            self.assertIs(bench.aggregate_precision_usage([qp], [9], settings)["tirust_qp_numerator_enabled"], value)


if __name__ == "__main__":
    unittest.main()
