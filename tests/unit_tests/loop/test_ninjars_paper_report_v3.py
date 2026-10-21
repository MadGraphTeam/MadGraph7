"""Paper table/figure semantics only; fixtures never constitute physical evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import ninjars_paper_report_v3 as report


class NinjaRSPaperReportV3Test(unittest.TestCase):
    def setUp(self):
        (report.ROOT / ".ninjars-build/tmp").mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=report.ROOT / ".ninjars-build/tmp", prefix="paper-report-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    @staticmethod
    def observations(deltas):
        return [{"point": i+1, "classification": "finite" if value is not None else "backend_rejected",
                 "delta": value} for i, value in enumerate(deltas)]

    def cases(self):
        return {"fixture": {"process": "SYNTHETIC software fixture", "N": 4, "group": "qcd",
            "sample_kind": "fixed software fixture", "helicity": "one synthetic helicity",
            "tools": {"1": {"status": "complete", "scores": {"combined": self.observations([None, 0, 2e-7, 1e-6])}},
                      "6": {"status": "complete", "scores": {"combined": self.observations([0, 0, 1e-8, 1e-6])}},
                      "8": {"status": "complete", "scores": {"combined": self.observations([0, 0, 0, 0])}}}}}

    def test_failed_points_and_all_zero_stay_in_denominator(self):
        rows = self.observations([None, 1e-8, 1e-7]) + [{"point": 4, "classification": "all_zero", "delta": 0}]
        result = report.tail(rows, [1e-8, 1e-7])
        self.assertEqual([row["k"] for row in result["rows"]], [2, 1])
        self.assertEqual(result["N"], 4)
        self.assertEqual(result["categories"]["all_zero"], 1)
        self.assertEqual(result["failure_or_unbounded"], 1)

    def test_nonfinite_and_unbounded_failures_count_at_every_threshold(self):
        rows = [{"point": i, "classification": category, "delta": None} for i, category in
                enumerate(("nonfinite", "unbounded_zero_mean", "missing", "process_failure"))]
        self.assertEqual([row["fraction"] for row in report.tail(rows, [1e-16, 1.])["rows"]], [1., 1.])
        for wrong in (float("nan"), float("inf"), -1.):
            with self.assertRaises(ValueError):
                report.tail(self.observations([wrong]), [1e-8])

    def test_zero_event_bound_is_not_empirical_fraction(self):
        result = report.tail(self.observations([0]*100), [1e-8])
        row = result["rows"][0]
        self.assertEqual((row["k"], row["fraction"], result["empirical_resolution"]), (0, 0., .01))
        self.assertAlmostEqual(row["zero_event_upper95"], 1-.05**.01, places=14)
        self.assertAlmostEqual(row["zero_event_upper95"], .0295130496070399, places=14)
        self.assertIn("descriptive", result["interval_scope"])

    def test_ninja_zero_ratios_are_null_and_positive_denominator_allows_zero(self):
        a = report.tail(self.observations([1e-7, 0]), [1e-8, 1e-6])
        b = report.tail(self.observations([0, 0]), [1e-8, 1e-6])
        values = report.ninja_ratios(a, b)
        self.assertEqual([v["ratio"] for v in values], [None, None])
        self.assertEqual([v["classification"] for v in values], ["unbounded_positive_over_0", "undefined_0_over_0"])
        self.assertEqual(report.ninja_ratios(b, a)[0]["ratio"], 0)

    def test_ratios_reject_changed_points_or_thresholds(self):
        original = report.tail(self.observations([1e-7, 0]), [1e-8])
        for field, value in (("point_ids", [2, 1]), ("rows", [{**original["rows"][0], "threshold": 2e-8}])):
            changed = copy.deepcopy(original); changed[field] = value
            with self.assertRaises(ValueError): report.ninja_ratios(original, changed)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            report.tail(self.observations([0]) * 2, [1e-8])

    def test_observed_jumps_enter_grid_and_absent_tool_has_no_curve(self):
        data = report.figure_data(self.cases(), [1e-9, 1e-5])["fixture"]
        self.assertEqual([r["threshold"] for r in data["tools"]["1"]["tail"]["rows"]],
                         [1e-9, 1e-8, 2e-7, 1e-6, 1e-5])
        self.assertIsNone(data["tools"]["7"]["tail"])
        self.assertEqual(data["tools"]["7"]["status"], "pending")
        self.assertEqual(data["declared_thresholds"], [1e-9, 1e-5])

    def inventory_fixture(self):
        output = self.directory / "output"; directory = output / "SubProcesses/P0_fixture"
        directory.mkdir(parents=True)
        files = {"loop_matrix.f": "      PARAMETER (NLOOPS=44,NLOOPGROUPS=28,NBORNAMPS=3,NCOMB=16)\n",
                 "loop_max_coefs.inc": "      PARAMETER (LOOPMAXCOEFS=70)\n",
                 "coef_construction_1.f": "\n".join("C Coefficient construction for loop diagram with ID %s" % i for i in range(1,37))}
        manifest = {}
        for name, value in files.items():
            target = directory / name; target.write_text(value)
            manifest[str(target.relative_to(output))] = {"sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "bytes": len(target.read_bytes())}
        identity = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        paths = {"output": output}
        for key, value in (("generation", {"seconds": 7.}), ("compilation", {"seconds": 3., "generated_sources_sha256": identity}),
                           ("compiled_manifest", {"files": manifest, "files_sha256": identity})):
            paths[key] = self.directory / (key+".json"); paths[key].write_text(json.dumps(value))
        paths["generation_log"] = self.directory / "generation.log"
        paths["generation_log"].write_text("Contributing diagrams generated: 3 Born, 36(+9) loops, 21 R2, 94 UV")
        return paths

    def test_loop_classes_are_not_helas_amplitudes_or_flavor_copies(self):
        inputs = {}; data = report.process_inventory(**self.inventory_fixture(), inputs=inputs)
        self.assertEqual((data["loop_diagram_classes"], data["helas_loop_amplitudes"], data["identified_equivalent_copies"]), (36, 44, 9))
        self.assertEqual((data["reducible_loop_groups"], data["maximum_tensor_rank"]), (28, 4))
        self.assertEqual(data["generation_cli_seconds"], 7.)
        self.assertEqual(len(inputs), 7)

    def test_actual_generated_qcd2_count_definitions_agree(self):
        release = report.ROOT / "docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/generation"
        output = report.ROOT / ".benchmark-work/policy-v3/final-pins-20261021-v1/generation/outputs/candidate/qcd-2"
        if not output.is_dir():
            self.skipTest("Actual final generated source is not retained in this checkout")
        metadata = release / "inputs/candidate/qcd-2"
        result = report.process_inventory(output, metadata / "generation.json", metadata / "compilation.json",
            metadata / "compiled-sources.json", release / "logs/generate-candidate-qcd-2.log")
        self.assertEqual(result["loop_diagram_classes"], result["cli_diagram_counts"]["loop_classes"])
        self.assertEqual((result["loop_diagram_classes"], result["helas_loop_amplitudes"],
                          result["reducible_loop_groups"]), (36, 44, 28))
        self.assertGreater(result["maximum_tensor_rank"], 0)

    def test_changed_source_manifest_digest_is_rejected(self):
        arguments = self.inventory_fixture()
        filename = arguments["compiled_manifest"]; data = json.loads(filename.read_text())
        data["files_sha256"] = "0"*64; filename.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "manifest identity"):
            report.process_inventory(**arguments)

    def test_changed_count_source_or_contradictory_log_is_rejected(self):
        arguments = self.inventory_fixture()
        arguments["generation_log"].write_text("Contributing diagrams generated: 3 Born, 35(+9) loops, 21 R2, 94 UV")
        with self.assertRaisesRegex(ValueError, "disagree"): report.process_inventory(**arguments)
        arguments["generation_log"].write_text("")
        (arguments["output"] / "SubProcesses/P0_fixture/loop_matrix.f").write_text("      PARAMETER(NLOOPS=45)")
        with self.assertRaisesRegex(ValueError, "Changed generated"): report.process_inventory(**arguments)

    def test_missing_count_evidence_stays_missing(self):
        arguments = self.inventory_fixture()
        arguments["generation_log"] = None
        directory = arguments["output"] / "SubProcesses/P0_fixture"
        (directory / "coef_construction_1.f").unlink(); (directory / "loop_max_coefs.inc").unlink()
        value = report.process_inventory(**arguments)
        self.assertIsNone(value["loop_diagram_classes"])
        self.assertIsNone(value["maximum_tensor_rank"])

    def test_parameter_filter_ignores_unrelated_local_parameters(self):
        source = "      PARAMETER(I=1,NLOOPS=44)\n      PARAMETER(I=2)\nC PARAMETER(NLOOPS=999)"
        self.assertEqual(report.parameters(source, {"NLOOPS"}), {"NLOOPS": 44})

    def test_cost_tables_preserve_failed_backend_counts_and_pending_numerator(self):
        observed = {"status": "failed_physics", "timing_repetitions": 8, "failed": 2, "dp": 798, "qp": 0,
                    "median_seconds_per_point": .5, "timed_status_counts": {"410": 2, "217": 798}}
        result = report.table_rows({"cases": {"qcd-4": {"backend": {"7": observed}}}}, {})["qcd-4"]
        self.assertEqual(result["single_helicity"]["backend"]["7"], observed)
        self.assertEqual(result["numerator_dp"]["status"], "pending")
        self.assertNotIn("reduction_seconds", result)

    def test_summary_is_revalidated_and_its_transitive_input_map_is_retained(self):
        filename = self.directory / "raw-manifest.json"; filename.write_text("{}")
        expected = {"runtime_identity_sha256": "a"*64}
        def reader(path, expected=None, inputs=None):
            self.assertEqual(expected, {"cohort": {"runtime_identity_sha256": "a"*64}})
            inputs.add(path)
            return {"schema": "ninjars-dp-stability-summary-v1", "stream": "forced_dp_self_consistency", "rescue_enabled": False}
        module = SimpleNamespace(__file__=report.__file__, report_snapshot=reader)
        with mock.patch.object(report.importlib, "import_module", return_value=module) as imported:
            value = report.build_snapshot(stability=filename, expected=expected)
        imported.assert_called_once_with("ninjars_dp_stability")
        self.assertIn(str(filename.relative_to(report.ROOT)), value["inputs_sha256"])
        self.assertFalse(value["qualification_eligible"])

    def test_rescued_summary_cannot_supply_dp_figures(self):
        filename = self.directory / "raw-manifest.json"; filename.write_text("{}")
        module = SimpleNamespace(__file__=report.__file__, report_snapshot=lambda *a, **k:
            {"schema": "ninjars-dp-stability-summary-v1", "stream": "normal", "rescue_enabled": True})
        with mock.patch.object(report.importlib, "import_module", return_value=module):
            with self.assertRaisesRegex(ValueError, "unrescued DP"): report.build_snapshot(stability=filename)

    def test_timing_reader_receives_exact_expected_runtime_and_map(self):
        filename = self.directory / "timing-result.json"; filename.write_text("{}")
        def reader(path, expected=None, inputs=None):
            self.assertEqual(expected, "a"*64)
            inputs[str(path.relative_to(report.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            return {"schema": "ninjars-paper-timing-summary-v1", "cases": {}}
        with mock.patch.object(report.importlib, "import_module", return_value=SimpleNamespace(
                __file__=report.__file__, report_snapshot=reader)):
            result = report.build_snapshot(timing=filename, expected={"runtime_identity_sha256": "a"*64})
        self.assertFalse(result["qualification_eligible"])

    def test_ratio_panels_cannot_use_truncated_tool_denominator(self):
        cases = self.cases(); cases["fixture"]["tools"]["1"]["scores"]["combined"].pop()
        with self.assertRaisesRegex(ValueError, "omitted a planned observation"):
            report.figure_data(cases, [1e-9, 1e-5])

    def test_nonsoftware_group_render_requires_physical_qualification(self):
        snapshot = {"schema": report.SCHEMA, "stability": {"stream": "forced_dp_self_consistency",
            "rescue_enabled": False, "physical_qualified": False, "observations_complete": True,
            "cases": self.cases(), "thresholds": [1e-9, 1e-5]}}
        with self.assertRaisesRegex(ValueError, "qualified runtime/process-parity"):
            report.plot_groups(snapshot, self.directory / "unqualified")
        self.assertFalse((self.directory / "unqualified").exists())
        with mock.patch.object(report, "plot_stability", return_value={"scope": "software stub"}) as render:
            report.plot_groups(snapshot, self.directory / "software", software_fixture=True)
            self.assertTrue(render.call_args.kwargs["software_fixture"])
            snapshot["stability"]["physical_qualified"] = True
            report.plot_groups(snapshot, self.directory / "qualified")
            self.assertFalse(render.call_args.kwargs["software_fixture"])

    def construction_fixture(self):
        timing = self.directory / "full-result.json"; timing.write_text("{}")
        numerator = self.directory / "construction-result.json"; numerator.write_text("{}")
        parent = {"path": str(timing.relative_to(report.ROOT)),
                  "sha256": hashlib.sha256(timing.read_bytes()).hexdigest()}
        total = {"schema": "ninjars-paper-timing-summary-v1", "source": parent,
                 "cases": {"fixture": {"backend": {"7": {"status": "failed_physics", "failed": 2}}}}}
        residual = {"median_difference_seconds_per_point": -1., "confidence_interval95": [-2., .5],
                    "bootstrap": "independent percentile; separate measurement sessions",
                    "label": "Remaining full-evaluation work, including stability/rescue; not pure reduction",
                    "ordinary_physics_status": "failed_physics"}
        case = {"scope": "DP construction diagnostic; not physical amplitude/stability/rescue acceptance",
                "median_seconds_per_point": 2., "confidence_interval95": [1.8, 2.1],
                "timing_repetitions": 8, "points": 100, "remaining": {"7": residual},
                "ordinary_physics_status": {"7": "failed_physics"}}
        construction = {"schema": "ninjars-paper-construction-summary-v1", "acceptance_eligible": False,
                        "ordinary_result": copy.deepcopy(parent), "cases": {"fixture": case}}
        return timing, numerator, total, construction

    def construction_readers(self, total, construction):
        modules = {}
        for name, value in (("ninjars_paper_timing_v3", total), ("ninjars_paper_numerator_v3", construction)):
            modules[name] = SimpleNamespace(__file__=str(report.ROOT / "tools" / (name+".py")),
                report_snapshot=mock.Mock(return_value=value))
        return modules

    def test_construction_requires_selected_timing_before_importing_reader(self):
        with mock.patch.object(report.importlib, "import_module") as imported:
            with self.assertRaisesRegex(ValueError, "selected full timing result"):
                report.build_snapshot(numerator=self.directory / "any.json")
            imported.assert_not_called()

    def test_construction_retains_negative_residual_ci_failed_status_and_source(self):
        timing, numerator, total, construction = self.construction_fixture()
        modules = self.construction_readers(total, construction)
        with mock.patch.object(report.importlib, "import_module", side_effect=lambda name: modules[name]):
            snapshot = report.build_snapshot(timing=timing, numerator=numerator,
                expected={"runtime_identity_sha256": "a"*64})
        cell = snapshot["tables"]["fixture"]
        self.assertEqual(cell["numerator_dp"], construction["cases"]["fixture"])
        self.assertEqual(cell["remaining"]["7"], construction["cases"]["fixture"]["remaining"]["7"])
        self.assertEqual(cell["remaining"]["7"]["median_difference_seconds_per_point"], -1.)
        self.assertEqual(cell["remaining"]["7"]["confidence_interval95"], [-2., .5])
        self.assertEqual(cell["remaining"]["7"]["ordinary_physics_status"], "failed_physics")
        self.assertFalse(snapshot["qualification_eligible"])
        modules["ninjars_paper_numerator_v3"].report_snapshot.assert_called_once()
        self.assertEqual(modules["ninjars_paper_numerator_v3"].report_snapshot.call_args.kwargs["expected"], "a"*64)
        for filename in (numerator, Path(modules["ninjars_paper_numerator_v3"].__file__)):
            self.assertEqual(snapshot["inputs_sha256"][str(filename.relative_to(report.ROOT))],
                             hashlib.sha256(filename.read_bytes()).hexdigest())

    def test_other_construction_parent_path_or_hash_cannot_populate_timing_table(self):
        timing, numerator, total, construction = self.construction_fixture()
        for key, value in (("path", "other-campaign/result.json"), ("sha256", "b"*64)):
            wrong = copy.deepcopy(construction); wrong["ordinary_result"][key] = value
            modules = self.construction_readers(total, wrong)
            with self.subTest(key=key), mock.patch.object(report.importlib, "import_module", side_effect=lambda name: modules[name]):
                with self.assertRaisesRegex(ValueError, "exact selected full timing result"):
                    report.build_snapshot(timing=timing, numerator=numerator)

    def test_construction_without_parent_binding_or_with_acceptance_claim_rejects(self):
        timing, numerator, total, construction = self.construction_fixture()
        for wrong in ({k:v for k,v in construction.items() if k != "ordinary_result"},
                      dict(construction, acceptance_eligible=True)):
            modules = self.construction_readers(total, wrong)
            with mock.patch.object(report.importlib, "import_module", side_effect=lambda name: modules[name]):
                with self.assertRaises(ValueError): report.build_snapshot(timing=timing, numerator=numerator)

    def test_missing_or_extra_construction_case_cannot_be_mixed_with_timing(self):
        timing, numerator, total, construction = self.construction_fixture()
        for cases in ({}, {**construction["cases"], "other": construction["cases"]["fixture"]}):
            modules = self.construction_readers(total, dict(construction, cases=cases))
            with mock.patch.object(report.importlib, "import_module", side_effect=lambda name: modules[name]):
                with self.assertRaisesRegex(ValueError, "cases must match"):
                    report.build_snapshot(timing=timing, numerator=numerator)

    def test_timing_reader_rejection_blocks_construction_import(self):
        timing, numerator, total, construction = self.construction_fixture()
        modules = self.construction_readers(total, construction)
        modules["ninjars_paper_timing_v3"].report_snapshot.side_effect = ValueError("ordinary source/runtime changed")
        with mock.patch.object(report.importlib, "import_module", side_effect=lambda name: modules[name]):
            with self.assertRaisesRegex(ValueError, "source/runtime changed"):
                report.build_snapshot(timing=timing, numerator=numerator)
        modules["ninjars_paper_numerator_v3"].report_snapshot.assert_not_called()

    def two_tool_cases(self):
        cases = self.cases()
        cases["fixture"]["requested_tools"] = [6, 8]
        cases["fixture"]["tools"].pop("1")
        return cases

    def test_declared_two_tool_curves_have_no_unrequested_cells(self):
        data = report.figure_data(self.two_tool_cases(), [1e-9, 1e-5], requested_tools=[6, 8])["fixture"]
        self.assertEqual(list(data["tools"]), ["6", "8"])
        self.assertEqual(data["requested_tools"], [6, 8])
        self.assertNotIn("1", data["tools"])
        self.assertNotIn("7", data["tools"])
        self.assertIsNotNone(data["tools"]["8"]["ratio"])

    def test_unattempted_requested_tool_remains_pending_without_extra_tools(self):
        cases = self.two_tool_cases(); cases["fixture"]["tools"].pop("8")
        data = report.figure_data(cases, [1e-9, 1e-5], requested_tools=[6, 8])["fixture"]
        self.assertEqual(list(data["tools"]), ["6", "8"])
        self.assertEqual(data["tools"]["8"]["status"], "pending")
        self.assertIsNone(data["tools"]["8"]["tail"])

    def test_invalid_reordered_or_noninteger_tool_selection_is_rejected(self):
        for selection in ([8, 6], [6], [6, 8, 7], [1, 8], [True, 8], [6., 8], ["6", "8"]):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                report.figure_data(self.two_tool_cases(), [1e-9, 1e-5], requested_tools=selection)

    def test_undeclared_tools_or_case_selection_cannot_enter_two_tool_plot(self):
        cases = self.two_tool_cases(); cases["fixture"]["tools"]["1"] = self.cases()["fixture"]["tools"]["1"]
        with self.assertRaisesRegex(ValueError, "Undeclared"):
            report.figure_data(cases, [1e-9, 1e-5], requested_tools=[6, 8])
        cases = self.two_tool_cases(); cases["fixture"]["requested_tools"] = [1, 6, 7, 8]
        with self.assertRaisesRegex(ValueError, "selections differ"):
            report.figure_data(cases, [1e-9, 1e-5], requested_tools=[6, 8])

    def test_group_render_propagates_validated_two_tool_selection(self):
        snapshot = {"schema": report.SCHEMA, "stability": {"stream": "forced_dp_self_consistency",
            "rescue_enabled": False, "physical_qualified": True, "requested_tools": [6, 8],
            "cases": self.two_tool_cases(), "thresholds": [1e-9, 1e-5]}}
        with mock.patch.object(report, "plot_stability", return_value={}) as render:
            report.plot_groups(snapshot, self.directory / "two-tools")
        self.assertEqual(render.call_args.kwargs["requested_tools"], [6, 8])
        self.assertFalse(render.call_args.kwargs["software_fixture"])

    def test_two_tool_svg_has_only_selected_legends(self):
        filename = self.directory / "two-tool-layout.svg"
        data = report.plot_stability(self.two_tool_cases(), [1e-9, 1e-5], filename,
            software_fixture=True, requested_tools=[6, 8])
        self.assertEqual(data["requested_tools"], [6, 8])
        svg = filename.read_text()
        self.assertIn("Ninja:", svg); self.assertIn("NinjaRS:", svg)
        self.assertNotIn("CutTools", svg); self.assertNotIn("COLLIER", svg)
        self.assertFalse(data["qualification_eligible"])

    def test_deterministic_synthetic_svg_labels_zeros_and_ratio_gaps(self):
        first, second = self.directory / "first.svg", self.directory / "second.svg"
        one = report.plot_stability(self.cases(), [1e-9, 1e-5], first, software_fixture=True)
        two = report.plot_stability(self.cases(), [1e-9, 1e-5], second, software_fixture=True)
        self.assertEqual(one["svg_sha256"], two["svg_sha256"])
        svg = first.read_text()
        for label in ("SOFTWARE FIXTURE, no physics", "gaps: Ninja 0/N", "one-sided 95%", "COLLIER (pending)"):
            self.assertIn(label, svg)
        self.assertFalse(one["qualification_eligible"])
        with self.assertRaisesRegex(ValueError, "Preserve"): report.plot_stability(self.cases(), [1e-9, 1e-5], first)


if __name__ == "__main__":
    unittest.main()
