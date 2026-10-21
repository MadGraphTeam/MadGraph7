"""Software aggregation fixtures; no CLI timing or physics measurements."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import ninjars_cli_report_v3 as report
from tools import ninjars_profile_cli_v3 as cli_authority


class NinjaRSCliReportV3Test(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=report.ROOT, prefix=".test-cli-report-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.expected = {name: value*64 for name, value in
            (("ninja_archive", "a"), ("ninjars_archive", "b"), ("process_checks", "c"))}
        self.specification = {"phase": "mg7-profile-cli-ninja-nrs-v1", "ready": True,
            "cases": {"qcd-2": {"process": "SYNTHETIC fixture", "model_name": "loop_sm", "nfinal": 2}},
            "schedule": [{"case": "qcd-2", "backend": backend, "repetition": rep, "npoints": 100 if rep == 0 else 0}
                for rep in range(7) for backend in (6, 8)]}
        self.results, self.records, self.failure_records = [], {}, {}
        for index, item in enumerate(self.specification["schedule"]):
            filename = self.directory / (str(index)+".json")
            filename.write_text(json.dumps({"software_fixture": True, "index": index}))
            self.results.append(filename)
            total = 1. if item["backend"] == 6 else .5
            self.records[filename] = {"schema": "validated-mg7-cli-profile-result-v1",
                "source": {"path": str(filename.relative_to(report.ROOT)), "sha256": hashlib.sha256(filename.read_bytes()).hexdigest()},
                "specification": {"path": "synthetic/specification.json", "sha256": "d"*64},
                "declared_specification": copy.deepcopy(self.specification), "item": copy.deepcopy(item),
                "acceptance_eligible": False, "timed_status_counts_available": False,
                "runtime_pins": copy.deepcopy(self.expected), "stage_scope_complete": True,
                "stage_scope": {"physical_card_sha256": "e"*64, "physical_helicity": [-1, -1, 1, 1],
                    "actual_split_orders_target": -1, "timing_energy_source": ["SQRTS=1000D0"],
                    "selected_helicity_weight": 2, "helicity_table_sha256": "4"*64,
                    "helicity_filter_sha256": "5"*64, "helicity_index": 1,
                    "loop_optimized_output": True, "direct_dp": True,
                    "common_madloop_settings": {"CTModeRun": 1}, "compiler_settings_sha256": "6"*64},
                "raw_timing": {mode: {"seconds_per_point": total, "construction_seconds_per_point": total*.1,
                    "reduction_estimate_seconds_per_point": total*.9} for mode in report.MODES},
                "timing": {"single_helicity": {"total": {"milliseconds": total*1000,
                    "printed_rounding_bound_ms": .5}}}, "stage_point_counts": {"boot": 0,
                    "polarized_full": 1000, "polarized_skip": 1000, "unpolarized_full": 500, "unpolarized_skip": 500},
                "selected_helicity_weight": 2, "runtime_files": {"fixture": {"sha256": "f"*64}},
                "generated_sources": "1"*64, "timing_metadata": {"path": "synthetic/timing.json", "sha256": "2"*64},
                "stability": {"dp_points": 100, "scope": "cache only; no common-point authority"},
                "new_stability_points": item["npoints"]}
        authority_file = self.directory / "authority.py"; authority_file.write_text("# SOFTWARE FIXTURE authority\n")
        def reader(filename, expected=None, inputs=None):
            self.assertEqual(expected, self.expected)
            inputs["synthetic/transitive-source"] = "3"*64
            return copy.deepcopy(self.records[filename])
        def failure_reader(filename, expected=None, inputs=None):
            self.assertEqual(expected, self.expected)
            inputs["synthetic/failed-command.log"] = "7"*64
            return copy.deepcopy(self.failure_records[filename])
        self.authority = SimpleNamespace(__file__=str(authority_file), report_snapshot=mock.Mock(side_effect=reader),
            failure_snapshot=mock.Mock(side_effect=failure_reader),
            same_stage_scope=mock.Mock(wraps=cli_authority.same_stage_scope))

    def build(self, results=None, failures=None):
        with mock.patch.object(report.importlib, "import_module", return_value=self.authority) as imported:
            value = report.build_snapshot(self.results if results is None else results, self.expected, draws=101,
                                          failures=failures)
        imported.assert_called_once_with("ninjars_profile_cli_v3")
        return value

    def failure(self, index=-1):
        original = self.records[self.results[index]]
        filename = self.directory / ("failed-"+str(index)+".json")
        filename.write_text(json.dumps({"software_fixture": True, "error": "retained failure"}))
        self.failure_records[filename] = {"schema": "retained-mg7-cli-profile-failure-v1", "status": "failed",
            "source": {"path": str(filename.relative_to(report.ROOT)), "sha256": hashlib.sha256(filename.read_bytes()).hexdigest()},
            "item": copy.deepcopy(original["item"]), "specification": copy.deepcopy(original["specification"]),
            "declared_specification": copy.deepcopy(original["declared_specification"]),
            "runtime_pins": copy.deepcopy(self.expected), "producer_source": {"path": "synthetic/producer.py", "sha256": "8"*64},
            "error": "SYNTHETIC RuntimeError: execution failed", "raw_files": {"failed-command.log":
                {"path": "synthetic/failed-command.log", "sha256": "7"*64}}}
        return filename

    def declare_sessions(self, repetitions=3, initial_stability_points=20):
        self.specification.update(repetitions=repetitions, initial_stability_points=initial_stability_points)
        self.specification["schedule"] = [{"case": "qcd-2", "backend": backend, "repetition": rep,
            "npoints": initial_stability_points if rep == 0 else 0}
            for rep in range(repetitions) for backend in (6, 8)]
        self.results = [filename for filename in self.results
                        if self.records[filename]["item"]["repetition"] < repetitions]
        for record in self.records.values():
            record["declared_specification"] = copy.deepcopy(self.specification)
            npoints = initial_stability_points if record["item"]["repetition"] == 0 else 0
            record["item"]["npoints"] = npoints
            record["new_stability_points"] = npoints
            record["stability"]["dp_points"] = initial_stability_points

    def test_concrete_authority_rechecks_every_result_and_binds_sources(self):
        result = self.build()
        self.assertEqual(self.authority.report_snapshot.call_count, 14)
        self.assertIn("synthetic/transitive-source", result["inputs_sha256"])
        for filename in (self.results[0], Path(self.authority.__file__), Path(report.__file__)):
            self.assertEqual(result["inputs_sha256"][str(filename.relative_to(report.ROOT))],
                hashlib.sha256(filename.read_bytes()).hexdigest())
        self.assertFalse(result["qualification_eligible"])
        self.assertFalse(result["fullsum_acceptance_eligible"])
        self.assertFalse(result["random_dp_stability_eligible"])

    def test_complete_series_preserves_two_modes_and_ratio_boundary(self):
        result = self.build(); case = result["cases"]["qcd-2"]
        self.assertTrue(result["complete"])
        self.assertEqual(case["sessions"], {"6": 7, "8": 7})
        self.assertEqual(case["status"], "complete_comparable")
        for mode in report.MODES:
            comparison = case["timings"][mode]["total"]["ninjars_over_ninja"]
            self.assertEqual(comparison["ratio"], .5)
            self.assertEqual(comparison["confidence_interval95"], [.5, .5])
            self.assertIn("independent-session", comparison["bootstrap"]["method"])
        self.assertIn("Full-minus-SKIP", result["reduction_estimate_scope"])
        self.assertFalse(result["timed_status_counts_available"])
        self.assertNotIn("qp", case)

    def test_missing_session_is_pending_without_a_paired_ratio(self):
        result = self.build(self.results[:-1]); case = result["cases"]["qcd-2"]
        self.assertFalse(result["complete"])
        self.assertEqual(case["sessions"], {"6": 7, "8": 6})
        self.assertEqual(case["status"], "pending")
        ratio = case["timings"]["helicity_sum"]["total"]["ninjars_over_ninja"]
        self.assertEqual(ratio["status"], "pending"); self.assertIsNone(ratio["ratio"])

    def test_unobserved_declared_case_keeps_exactly_two_pending_tools(self):
        for record in self.records.values():
            specification = record["declared_specification"]
            specification["cases"]["loop-h"] = {"process": "SYNTHETIC loop fixture", "nfinal": 1}
            specification["schedule"].extend({"case": "loop-h", "backend": backend, "repetition": rep,
                "npoints": 100 if rep == 0 else 0} for rep in range(7) for backend in (6, 8))
        case = self.build()["cases"]["loop-h"]
        self.assertEqual(case["sessions"], {"6": 0, "8": 0})
        self.assertEqual(case["status"], "pending")
        self.assertEqual(set(case["observations"]), {"6", "8"})

    def test_duplicate_session_rejects(self):
        with self.assertRaisesRegex(ValueError, "Duplicate CLI"):
            self.build(self.results + self.results[:1])

    def test_mixed_specification_content_or_binding_rejects(self):
        for field, changed in (("specification", {"path": "other/spec.json", "sha256": "d"*64}),
                               ("declared_specification", {**self.specification, "ready": False})):
            original = self.records[self.results[-1]][field]
            self.records[self.results[-1]][field] = changed
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "different study"):
                self.build()
            self.records[self.results[-1]][field] = original

    def test_explicit_expected_pins_required_before_import(self):
        for expected in (None, {}, {**self.expected, "runtime_identity_sha256": "d"*64},
                         {**self.expected, "process_checks": "BAD"}):
            with self.subTest(expected=expected), mock.patch.object(report.importlib, "import_module") as imported:
                with self.assertRaisesRegex(ValueError, "exact Ninja/NinjaRS"):
                    report.build_snapshot(self.results, expected)
                imported.assert_not_called()

    def test_runtime_pin_mismatch_rejects_even_validated_label(self):
        self.records[self.results[0]]["runtime_pins"]["ninjars_archive"] = "9"*64
        with self.assertRaisesRegex(ValueError, "runtime/source pins"):
            self.build()

    def test_changed_source_helicity_split_or_energy_makes_stage_incomparable(self):
        for key, changed in (("physical_card_sha256", "9"*64), ("physical_helicity", [1, 1, 1, 1]),
                             ("actual_split_orders_target", 2), ("timing_energy_source", ["SQRTS=500D0"]),
                             ("selected_helicity_weight", 4), ("helicity_table_sha256", "9"*64),
                             ("helicity_filter_sha256", "8"*64)):
            original = copy.deepcopy(self.records[self.results[-1]]["stage_scope"])
            self.records[self.results[-1]]["stage_scope"][key] = changed
            with self.subTest(key=key):
                result = self.build(); case = result["cases"]["qcd-2"]
                self.assertFalse(result["complete"])
                self.assertEqual(case["status"], "complete_incomparable")
                self.assertEqual(case["timings"]["helicity_sum"]["total"]["ninjars_over_ninja"]["status"],
                    "incomparable_stage_scope")
            self.records[self.results[-1]]["stage_scope"] = original

    def test_missing_scope_metadata_never_receives_a_ratio(self):
        self.records[self.results[3]]["stage_scope_complete"] = False
        result = self.build(); case = result["cases"]["qcd-2"]
        self.assertEqual(case["status"], "complete_incomparable")
        self.assertIsNone(case["timings"]["single_helicity"]["total"]["ninjars_over_ninja"]["ratio"])

    def test_adaptive_stage_counts_may_differ_but_remain_explicit(self):
        record = self.records[self.results[-1]]
        record["stage_point_counts"]["polarized_full"] = 2000
        result = self.build(); case = result["cases"]["qcd-2"]
        self.assertTrue(result["complete"])
        self.assertEqual(case["observations"]["8"][-1]["stage_point_counts"]["polarized_full"], 2000)
        self.assertEqual(case["observations"]["6"][0]["stage_point_counts"]["polarized_full"], 1000)
        self.assertEqual(case["observations"]["8"][-1]["selected_helicity_weight"], 2)

    def test_signed_reduction_observations_are_not_clamped_or_colorable(self):
        for record in self.records.values():
            if record["item"]["backend"] == 8:
                record["raw_timing"]["helicity_sum"]["reduction_estimate_seconds_per_point"] = -.25
        case = self.build()["cases"]["qcd-2"]
        reduction = case["timings"]["helicity_sum"]["reduction_estimate"]
        self.assertEqual(reduction["backend"]["8"]["median_seconds_per_point"], -.25)
        self.assertEqual(reduction["backend"]["8"]["confidence_interval95"], [-.25, -.25])
        self.assertEqual(reduction["ninjars_over_ninja"]["status"], "nonpositive_samples")
        self.assertIsNone(reduction["ninjars_over_ninja"]["ratio"])
        self.assertEqual(case["timings"]["helicity_sum"]["total"]["ninjars_over_ninja"]["ratio"], .5)

    def test_printed_rounding_and_raw_float_timing_remain_separate(self):
        record = self.records[self.results[0]]
        record["raw_timing"]["single_helicity"]["seconds_per_point"] = 1.000123
        observation = self.build()["cases"]["qcd-2"]["observations"]["6"][0]
        self.assertEqual(observation["raw_timing"]["single_helicity"]["seconds_per_point"], 1.000123)
        self.assertEqual(observation["timing"]["single_helicity"]["total"]["printed_rounding_bound_ms"], .5)
        self.assertEqual(observation["new_stability_points"], 100)

    def test_reader_failure_cannot_be_promoted_to_a_summary(self):
        self.authority.report_snapshot.side_effect = ValueError("retained raw/source validation failed")
        with self.assertRaisesRegex(ValueError, "raw/source"):
            self.build()

    def test_complete_or_acceptance_labels_cannot_replace_raw_authority_schema(self):
        for key, changed in (("schema", "unvalidated-complete"), ("acceptance_eligible", True),
                             ("timed_status_counts_available", True)):
            original = self.records[self.results[0]][key]; self.records[self.results[0]][key] = changed
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "concrete diagnostic-only"):
                self.build()
            self.records[self.results[0]][key] = original

    def test_schedule_rejects_other_tools_wrong_cache_or_missing_sessions(self):
        for change in (lambda s: s["schedule"][0].update(backend=1),
                       lambda s: s["schedule"][0].update(npoints=0),
                       lambda s: s["schedule"].pop(),
                       lambda s: s["schedule"].append(s["schedule"][0])):
            specification = copy.deepcopy(self.specification); change(specification)
            with self.subTest(change=change), self.assertRaises(ValueError):
                report.validate_schedule(specification)

    def test_changed_item_not_in_declared_schedule_rejects(self):
        self.records[self.results[0]]["item"]["npoints"] = 0
        with self.assertRaisesRegex(ValueError, "Undeclared"):
            self.build()

    def test_observed_cell_type_coercion_cannot_match_the_declared_schedule(self):
        for key, changed in (("backend", 6.), ("repetition", False), ("npoints", 100.)):
            original = self.records[self.results[0]]["item"][key]
            self.records[self.results[0]]["item"][key] = changed
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "cell types"):
                self.build()
            self.records[self.results[0]]["item"][key] = original

    def test_statistics_are_deterministic_and_keep_mad_and_nonfinite_rejection(self):
        one = report.summarize([1., 2., 3., 4., 5., 6., 7.], draws=101)
        self.assertEqual(one, report.summarize([1., 2., 3., 4., 5., 6., 7.], draws=101))
        self.assertEqual(one["median_seconds_per_point"], 4.)
        self.assertEqual(one["median_absolute_deviation"], 2.)
        self.assertEqual(report.percentile([0., 10.], .25), 2.5)
        for values in ([], [float("nan")], [float("inf")], [True]):
            with self.subTest(values=values), self.assertRaises(ValueError): report.summarize(values)

    def test_failed_attempt_is_separate_from_unattempted_and_never_becomes_timing(self):
        filename = self.failure()
        result = self.build(self.results[:-1], failures=[filename]); case = result["cases"]["qcd-2"]
        self.assertEqual(result["retained_failed_attempts"], 1)
        self.assertEqual(case["status"], "failed_attempt")
        self.assertEqual(case["sessions"], {"6": 7, "8": 6})
        self.assertEqual(case["session_states"]["8"][-1]["state"], "failed_attempt")
        self.assertEqual(case["timings"]["helicity_sum"]["total"]["ninjars_over_ninja"]["status"], "failed_attempt")
        self.assertEqual(case["timings"]["helicity_sum"]["total"]["backend"]["8"]["sessions"], 6)
        failure = case["failed_attempts"]["8"][0]
        self.assertIn("SYNTHETIC RuntimeError", failure["error"])
        self.assertEqual(failure["raw_files"], self.failure_records[filename]["raw_files"])
        self.assertNotIn("timing", failure); self.assertNotIn("raw_timing", failure)
        self.assertIn("synthetic/failed-command.log", result["inputs_sha256"])
        self.assertIn("controller launch ledger", result["attempt_inventory_scope"])

    def test_all_failed_selection_still_has_declared_failed_and_unattempted_cells(self):
        filename = self.failure()
        case = self.build([], failures=[filename])["cases"]["qcd-2"]
        self.assertEqual(case["sessions"], {"6": 0, "8": 0})
        self.assertEqual(case["session_states"]["6"][0]["state"], "unattempted")
        self.assertEqual(case["session_states"]["8"][-1]["state"], "failed_attempt")
        self.assertEqual(case["timings"]["single_helicity"]["total"]["backend"]["8"]["status"], "failed_attempt")
        self.assertEqual(case["timings"]["single_helicity"]["total"]["backend"]["6"]["status"], "pending")

    def test_later_valid_observation_does_not_hide_retained_failed_attempt(self):
        filename = self.failure()
        result = self.build(failures=[filename]); case = result["cases"]["qcd-2"]
        self.assertTrue(result["complete"])
        self.assertEqual(case["session_states"]["8"][-1]["state"], "valid_with_retained_failures")
        self.assertEqual(len(case["failed_attempts"]["8"]), 1)
        self.assertEqual(case["sessions"]["8"], 7)
        self.assertFalse(result["qualification_eligible"])

    def test_duplicate_or_foreign_failed_descriptor_rejects(self):
        filename = self.failure()
        with self.assertRaisesRegex(ValueError, "Duplicate retained"):
            self.build(failures=[filename, filename])
        failure = self.failure_records[filename]
        failure["specification"]["sha256"] = "9"*64
        with self.assertRaisesRegex(ValueError, "different study"):
            self.build(failures=[filename])
        failure["specification"]["sha256"] = "d"*64
        failure["runtime_pins"]["ninjars_archive"] = "9"*64
        with self.assertRaisesRegex(ValueError, "runtime/source pins"):
            self.build(failures=[filename])

    def test_failed_descriptor_cannot_smuggle_a_timer_or_empty_error(self):
        filename = self.failure(); failure = self.failure_records[filename]
        for key, value in (("raw_timing", {"total": 1.}), ("timing", {}), ("acceptance_eligible", True), ("error", "")):
            original = failure.get(key); failure[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "without timing promotion"):
                self.build(failures=[filename])
            if original is None: failure.pop(key)
            else: failure[key] = original

    def test_failure_authority_rejection_is_not_relabelled_pending(self):
        filename = self.failure()
        self.authority.failure_snapshot.side_effect = ValueError("failed descriptor inventory changed")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            self.build(failures=[filename])

    def test_explicit_three_sessions_use_six_observations_and_twenty_initial_points(self):
        self.declare_sessions()
        result = self.build(); case = result["cases"]["qcd-2"]
        self.assertTrue(result["complete"])
        self.assertEqual(result["declared_scope"], {"repetitions": 3, "initial_stability_points": 20})
        self.assertEqual(self.authority.report_snapshot.call_count, 6)
        self.assertEqual(case["sessions"], {"6": 3, "8": 3})
        self.assertEqual(case["required_sessions_per_backend"], 3)
        for backend in ("6", "8"):
            self.assertEqual(len(case["session_states"][backend]), 3)
            self.assertEqual([record["new_stability_points"] for record in case["observations"][backend]], [20, 0, 0])
            self.assertEqual(case["timings"]["helicity_sum"]["total"]["backend"][backend]["sessions"], 3)
        self.assertIn("3 declared independent", result["uncertainty_scope"])
        self.assertFalse(result["fullsum_acceptance_eligible"])

    def test_three_session_uncertainty_uses_actual_samples_without_padding(self):
        self.declare_sessions()
        values = {6: [1., 2., 4.], 8: [.4, 1., 2.]}
        for filename in self.results:
            record = self.records[filename]
            backend, repetition = record["item"]["backend"], record["item"]["repetition"]
            for mode in report.MODES:
                record["raw_timing"][mode]["seconds_per_point"] = values[backend][repetition]
        timing = self.build()["cases"]["qcd-2"]["timings"]["helicity_sum"]["total"]
        for backend in (6, 8):
            summary = timing["backend"][str(backend)]
            self.assertEqual(summary, report.summarize(values[backend], draws=101))
            self.assertEqual(len(summary["observed_seconds_per_point"]), 3)
            self.assertGreater(summary["confidence_interval95"][1]-summary["confidence_interval95"][0], 0)
        self.assertEqual(timing["ninjars_over_ninja"], report.independent_ratio(values[6], values[8], draws=101))

    def test_three_session_missing_and_failed_cells_are_not_filled(self):
        self.declare_sessions()
        pending = self.build(self.results[:-1])["cases"]["qcd-2"]
        self.assertEqual(pending["sessions"], {"6": 3, "8": 2})
        self.assertEqual(pending["session_states"]["8"][-1]["state"], "unattempted")
        self.assertIsNone(pending["timings"]["single_helicity"]["total"]["ninjars_over_ninja"]["ratio"])
        failure = self.failure()
        failed = self.build(self.results[:-1], failures=[failure])["cases"]["qcd-2"]
        self.assertEqual(failed["status"], "failed_attempt")
        self.assertEqual(failed["sessions"], {"6": 3, "8": 2})
        self.assertEqual(len(failed["session_states"]["8"]), 3)
        self.assertEqual(failed["session_states"]["8"][-1]["state"], "failed_attempt")
        self.assertEqual(failed["timings"]["helicity_sum"]["total"]["backend"]["8"]["sessions"], 2)
        recovered = self.build(failures=[failure])["cases"]["qcd-2"]
        self.assertEqual(recovered["status"], "complete_comparable")
        self.assertEqual(recovered["session_states"]["8"][-1]["state"], "valid_with_retained_failures")
        self.assertEqual(len(recovered["failed_attempts"]["8"]), 1)

    def test_three_results_cannot_infer_a_three_session_study(self):
        subset = [filename for filename in self.results if self.records[filename]["item"]["repetition"] < 3]
        result = self.build(subset); case = result["cases"]["qcd-2"]
        self.assertFalse(result["complete"])
        self.assertEqual(result["declared_scope"], {"repetitions": 7, "initial_stability_points": 100})
        self.assertEqual(case["required_sessions_per_backend"], 7)
        self.assertEqual(case["sessions"], {"6": 3, "8": 3})
        self.assertEqual(len(case["session_states"]["8"]), 7)
        self.assertIsNone(case["timings"]["helicity_sum"]["total"]["ninjars_over_ninja"]["ratio"])

    def test_declared_session_and_initial_cache_fields_are_strictly_typed(self):
        for field, invalid in (("repetitions", [True, False, 3., "3", 0, 1, 2, None]),
                               ("initial_stability_points", [True, 20., "20", 0, 19, 21, 99, None])):
            for value in invalid:
                specification = copy.deepcopy(self.specification)
                specification[field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "three declared sessions"):
                    report.validate_schedule(specification)

    def test_declared_three_scope_rejects_extra_missing_or_wrong_cache_schedule(self):
        self.declare_sessions()
        for change in (lambda s: s["schedule"].pop(),
                       lambda s: s["schedule"].append({"case": "qcd-2", "backend": 6, "repetition": 3, "npoints": 0}),
                       lambda s: s["schedule"][0].update(npoints=100),
                       lambda s: s["schedule"][2].update(npoints=20),
                       lambda s: s.pop("initial_stability_points"),
                       lambda s: s.pop("repetitions")):
            specification = copy.deepcopy(self.specification)
            change(specification)
            with self.subTest(change=change), self.assertRaises(ValueError):
                report.validate_schedule(specification)

    def test_explicit_session_counts_above_three_keep_the_exact_declared_count(self):
        self.declare_sessions(repetitions=4, initial_stability_points=100)
        result = self.build(); case = result["cases"]["qcd-2"]
        self.assertTrue(result["complete"])
        self.assertEqual(case["sessions"], {"6": 4, "8": 4})
        self.assertEqual(case["required_sessions_per_backend"], 4)
        self.assertEqual(len(case["observations"]["6"]), 4)

    def test_three_session_all_failed_inventory_keeps_only_three_declared_slots(self):
        self.declare_sessions()
        failure = self.failure()
        result = self.build([], failures=[failure]); case = result["cases"]["qcd-2"]
        self.assertFalse(result["complete"])
        self.assertEqual(case["required_sessions_per_backend"], 3)
        self.assertEqual(case["sessions"], {"6": 0, "8": 0})
        self.assertEqual([row["state"] for row in case["session_states"]["8"]],
                         ["unattempted", "unattempted", "failed_attempt"])
        self.assertEqual(case["timings"]["helicity_sum"]["total"]["backend"]["8"],
                         {"status": "failed_attempt", "sessions": 0})

    def test_scope_defaults_and_light_study_match_the_concrete_adapter(self):
        for specification in [self.specification,
                {**self.specification, "repetitions": 3, "initial_stability_points": 20},
                {**self.specification, "repetitions": 4, "initial_stability_points": 100}]:
            with self.subTest(scope=specification.get("repetitions", 7)):
                self.assertEqual(report.declared_scope(specification), cli_authority.declared_scope(specification))


if __name__ == "__main__":
    unittest.main()
