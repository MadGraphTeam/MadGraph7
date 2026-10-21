"""DP construction scope and structural contracts; no numerical engines."""
import json
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_paper_numerator_v3 as num


class ConstructionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT, prefix=".test-numerator-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def raw(self, code=110, accuracy=-1., loops=1, directory=None):
        directory = directory or self.root
        (directory / "timing.dat").write_text("1.0 2.0\n")
        (directory / "dimensions.dat").write_text("1 1 0\n")
        (directory / "values.dat").write_text("".join(
            "%d %d %d 0.0 2.0 0.0 0.0 %s\n" % (p, o, code, accuracy)
            for p in range(1, 101) for o in (0, 1)))
        (directory / "timed_status.dat").write_text("%d %d\n" % (code, loops*100))

    def cards(self):
        card = num.MadLoopParam(str(ROOT / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
        card["MLReductionLib"] = "6"; card["CTModeRun"] = -1
        card["COLLIERComputeUVpoles"] = True; card["COLLIERComputeIRpoles"] = True
        card["MLStabThres"] = 1e-9; card["COLLIERRequiredAccuracy"] = 1e-12
        path = self.root / "Cards/MadLoopParams.dat"; path.parent.mkdir(); card.write(str(path))
        other = self.root / "SubProcesses/MadLoop5_resources/MadLoopParams.dat"; other.parent.mkdir(parents=True)
        other.symlink_to(path)
        return path, other

    def test_exactly_one_private_skip_declaration(self):
        source = "before\n      PARAMETER (SKIPLOOPEVAL=.FALSE.)\nafter\n"
        self.assertEqual(num.skip_reduction(source), source.replace(".FALSE.", ".TRUE."))
        for wrong in ("unrelated", source + source, source.replace(".FALSE.", ".TRUE.")):
            with self.assertRaises(ValueError): num.skip_reduction(wrong)

    def test_driver_changes_only_force_flag(self):
        ordinary = num.full.helicity_worker((ROOT / "tools/ninjars_benchmark_worker.f").read_text(), 10)
        construction = num.construction_driver(ordinary)
        self.assertEqual(construction.replace("FORCE_STABILITY_CHECK(.FALSE.)", "FORCE_STABILITY_CHECK(.TRUE.)"), ordinary)
        self.assertEqual(construction.count("SLOOPMATRIXHEL_THRES"), 3)
        for wrong in (ordinary.replace("SLOOPMATRIXHEL_THRES", "OTHER", 1), construction):
            with self.assertRaises(ValueError): num.construction_driver(wrong)

    def test_card_changes_only_direct_mode_and_filter_doublecheck(self):
        path, other = self.cards()
        before = num.configure_cards(self.root)
        for target in (path, other):
            actual = {str(k).lower(): v for k,v in dict(num.MadLoopParam(str(target))).items()}
            self.assertEqual(actual, {**before, "ctmoderun": 1, "doublecheckhelicityfilter": False})
            self.assertTrue(actual["colliercomputeuvpoles"] and actual["colliercomputeirpoles"])
            self.assertEqual(actual["mlstabthres"], 1e-9)

    def test_different_effective_card_is_not_repaired(self):
        _, other = self.cards(); other.unlink()
        card = num.MadLoopParam(str(self.root / "Cards/MadLoopParams.dat"))
        card["CTModeRun"] = 4; card.write(str(other))
        with self.assertRaisesRegex(ValueError, "Effective ordinary"): num.configure_cards(self.root)

    def test_accept_direct110_as_diagnostic_never_physics(self):
        self.raw(loops=3)
        row = num.parse_outputs(self.root, 100, 3)
        self.assertEqual(row["timed_status_counts"], {"110": 300})
        self.assertFalse(row["physical_acceptance_applicable"])
        self.assertTrue(row["diagnostic_complete"])
        self.assertNotIn("physical_valid", row)

    def test_normal_qp_initialization_and_failed_statuses_are_not_construction(self):
        for code in (216, 218, 322, 120, 130, 410):
            self.raw(code)
            with self.subTest(code=code), self.assertRaises(ValueError): num.parse_outputs(self.root, 100, 1)

    def test_accuracy_must_remain_unassessed(self):
        self.raw(accuracy=1e-12)
        with self.assertRaisesRegex(ValueError, "no stability estimate"): num.parse_outputs(self.root, 100, 1)

    def test_all_points_orders_and_real_counts_required(self):
        self.raw(); path = self.root / "values.dat"; path.write_text(path.read_text().split("\n", 1)[1])
        with self.assertRaises(ValueError): num.parse_outputs(self.root, 100, 1)
        self.raw(); (self.root / "timed_status.dat").write_text("110 99\n")
        with self.assertRaises(ValueError): num.parse_outputs(self.root, 100, 1)

    def test_nonfinite_raw_is_not_summarized(self):
        self.raw(); path = self.root / "values.dat"; path.write_text(path.read_text().replace("2.0", "nan", 1))
        with self.assertRaises(ValueError): num.parse_outputs(self.root, 100, 1)

    def test_remaining_is_independent_and_not_clamped(self):
        result = num.remaining([1.] * 8, [2.] * 8)
        self.assertEqual(result["median_difference_seconds_per_point"], -1.)
        self.assertEqual(result["confidence_interval95"], [-1., -1.])
        self.assertIn("not pure reduction", result["label"])
        self.assertIn("independent", result["bootstrap"])

    def test_calibration_does_not_replace_eighth_sample(self):
        for left, right in (([1.] * 7, [1.] * 8), ([1.] * 8, [1.] * 7), ([1.] * 8, [float('inf')] * 8)):
            with self.assertRaises(ValueError): num.remaining(left, right)

    def test_requires_actual_completed_ordinary_series(self):
        with mock.patch.object(num.full, "report_snapshot", side_effect=ValueError("not eight complete blocks")) as check:
            with self.assertRaisesRegex(ValueError, "eight complete"): num.make_spec(self.root / "calibration.json", self.root / "report", self.root / "work")
            check.assert_called_once()

    def test_no_authorization_cannot_reach_worker(self):
        with mock.patch.object(num.workers, "Worker") as worker:
            with self.assertRaises(ValueError): num.collect(self.root / "none", self.root / "none", False)
            worker.assert_not_called()
        path = self.root / "spec.json"; num.bench.dump(path, {})
        handoff = self.root / "handoff.json"; num.bench.dump(handoff, {})
        with mock.patch.object(num, "validate_spec", return_value={}), mock.patch.object(num.workers, "Worker") as worker:
            with self.assertRaises(ValueError): num.collect(path, handoff, True)
            worker.assert_not_called()

    def test_source_drift_blocks_clone_before_build(self):
        with mock.patch.object(num.bench, "generated_source_manifest", return_value={}), mock.patch.object(num.references, "clone_output") as clone:
            with self.assertRaises(ValueError): num.prepare({"output": str(self.root), "state": {"compiled_manifest": {"other": 1}}}, {}, self.root / "copy", self.root / "report")
            clone.assert_not_called()

    def test_every_raw_file_must_belong_to_result_inventory(self):
        self.raw()
        capture = {name: num.supplement.bound(self.root / name) for name in (*num.workers.OUTPUTS, "dimensions.dat")}
        inventory = {record["path"]: record["sha256"] for record in capture.values()}
        num.checked_raw_capture(capture, self.root, inventory)
        for missing in capture.values():
            malformed = dict(inventory); del malformed[missing["path"]]
            with self.subTest(path=missing["path"]), self.assertRaisesRegex(ValueError, "missing from result inventory"):
                num.checked_raw_capture(capture, self.root, malformed)
        malformed = dict(inventory); malformed[capture["values.dat"]["path"]] = "0" * 64
        with self.assertRaisesRegex(ValueError, "missing from result inventory"):
            num.checked_raw_capture(capture, self.root, malformed)

    def test_raw_inventory_does_not_allow_another_command_path(self):
        self.raw()
        capture = {name: num.supplement.bound(self.root / name) for name in (*num.workers.OUTPUTS, "dimensions.dat")}
        inventory = {record["path"]: record["sha256"] for record in capture.values()}
        with self.assertRaisesRegex(ValueError, "other raw reused"):
            num.checked_raw_capture(capture, self.root / "another-command", inventory)

    def test_summary_exposes_validated_ordinary_result_authority(self):
        report = self.root / "report"; report.mkdir()
        parent = self.root / "ordinary.json"; parent.write_text("{}")
        spec = report / "input-spec.json"; num.bench.dump(spec, {"ordinary_result": num.supplement.bound(parent)})
        num.bench.dump(report / "status.json", {"status": "complete", "requested": [], "completed": []})
        result = report / "result.json"
        num.bench.dump(result, num.policy.seal({"schema": num.RUN, "acceptance_eligible": False,
            "files": {}, "specification": num.supplement.bound(spec), "report": num.policy.relative(report)}))
        with mock.patch.object(num, "validate_spec", return_value={"cases": {}}), mock.patch.object(num.full, "report_snapshot", return_value={}) as validate:
            summary = num.report_snapshot(result)
        validate.assert_called_once_with(parent, expected=None, inputs=None)
        self.assertEqual(summary["ordinary_result"], num.supplement.bound(parent))
        self.assertFalse(summary["acceptance_eligible"])

    def collection_fixture(self, fail_sequence=None, startup_failure=False):
        report, work = self.root / "report", self.root / "work"
        points = self.root / "points.dat"; points.write_text("mock hundred-point transport\n")
        parent = self.root / "ordinary.json"; parent.write_text("{}")
        case = {"definition": {"id": "mock"}, "points": num.supplement.bound(points),
            "helicity_index": 10, "physical_helicity": [-1, 1, -1], "origin": {"reducer_runtime": {}}}
        spec = {"ordinary_result": num.supplement.bound(parent), "report": num.policy.relative(report),
            "work": num.policy.relative(work), "sources": {}, "timeout": 1.,
            "parents": {"mock": {"filter": {"sha256": "matched"}}}}
        spec_path = self.root / "spec.json"; num.bench.dump(spec_path, spec)
        handoff = self.root / "handoff.json"; num.bench.dump(handoff, {"root_authorized": True,
            "phase": num.RUN, "cpu": 207, "specification_sha256": num.workers.digest(spec_path)})
        instances, runtime_exit = [], []
        owner = self

        class SyntheticWorker:
            def __init__(self, command, directory, log, **kwargs):
                self.sequence = 0; self.points = 100; self.directory = directory; self.closed = False
                instances.append(self)
                if startup_failure: raise RuntimeError("injected constructor failure")

            def run(self, loops):
                self.sequence += 1; owner.raw(loops=loops, directory=self.directory)
                if self.sequence == fail_sequence: raise RuntimeError("injected RUN failure")
                return 1.

            def close(self): self.closed = True

        def prepare(parent, case, output, cell):
            directory = output / "SubProcesses/P0_mock"; directory.mkdir(parents=True)
            return directory, {"ready_for_numerical": True, "state": {}}

        @contextmanager
        def runtime(cell):
            try: yield
            finally: runtime_exit.append(str(cell))

        patches = ExitStack()
        for target, name, replacement in ((num, "validate_spec", mock.Mock(return_value={"cases": {"mock": case}})),
                (num.full, "report_snapshot", mock.Mock(return_value={})), (num, "prepare", prepare),
                (num.full, "target_state", mock.Mock(return_value={})),
                (num, "filter_record", mock.Mock(return_value={"sha256": "matched"})),
                (num.workers, "Worker", SyntheticWorker), (num.ownership, "owned_runtime", runtime),
                (num.workers, "isolated_harness", lambda *args: nullcontext()),
                (num, "report_snapshot", mock.Mock(return_value={"synthetic": True})),
                (num.os, "sched_getaffinity", mock.Mock(return_value={207}))):
            patches.enter_context(mock.patch.object(target, name, replacement))
        patches.enter_context(mock.patch.dict(num.os.environ, {name: "1" for name in num.pair.THREADS}))
        self.addCleanup(patches.close)
        return spec_path, handoff, report, instances, runtime_exit

    def test_mock_collection_keeps_calibration_separate_and_closes_worker(self):
        spec, handoff, report, instances, exits = self.collection_fixture()
        self.assertEqual(num.collect(spec, handoff, True), {"synthetic": True})
        self.assertEqual(len(instances), 1); self.assertEqual(instances[0].sequence, 9)
        self.assertTrue(instances[0].closed); self.assertEqual(len(exits), 1)
        data = report / "cases/mock/data"
        self.assertEqual({p.name for p in data.glob("*.json")}, {str(r) + ".json" for r in ("calibration", *range(8))})
        self.assertEqual(json.loads((report / "status.json").read_text())["status"], "complete")
        self.assertTrue((report / "result.json").is_file())

    def test_mock_run_failure_retains_raw_before_parsing_and_closes_worker(self):
        spec, handoff, report, instances, exits = self.collection_fixture(fail_sequence=3)
        with self.assertRaisesRegex(RuntimeError, "RUN failure"): num.collect(spec, handoff, True)
        self.assertTrue(instances[0].closed); self.assertEqual(len(exits), 1)
        raw = report / "cases/mock/raw/mock-candidate-6/construction-1"
        self.assertEqual({p.name for p in raw.iterdir()}, {*num.workers.OUTPUTS, "dimensions.dat"})
        self.assertEqual(json.loads((report / "status.json").read_text())["status"], "failed")
        self.assertFalse((report / "result.json").exists())

    def test_mock_startup_failure_exits_owned_context_without_measurement(self):
        spec, handoff, report, instances, exits = self.collection_fixture(startup_failure=True)
        with self.assertRaisesRegex(RuntimeError, "constructor failure"): num.collect(spec, handoff, True)
        self.assertEqual(len(exits), 1)
        self.assertEqual(instances[0].sequence, 0)
        self.assertEqual(json.loads((report / "status.json").read_text())["status"], "failed")
        self.assertFalse((report / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
