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
