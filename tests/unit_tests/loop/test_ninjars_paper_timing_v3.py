"""Paper-timing contracts and failed-physics evidence; no numerical engines."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_paper_timing_v3 as paper


class PaperTimingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".benchmark-work", prefix="paper-timing-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def raw(self, directory, code=218, loops=1, accuracy=1e-12, seconds=2.):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "timing.dat").write_text("1.0 %s\n" % seconds)
        (directory / "dimensions.dat").write_text("1 1 1\n")
        (directory / "values.dat").write_text("".join(
            "%d %d %d 1.0 2.0 3.0 4.0 %s\n" % (p, o, code, accuracy)
            for p in range(1, 101) for o in range(2)))
        (directory / "timed_status.dat").write_text("%d %d\n" % (code, loops * 100))

    def minimal_spec(self):
        return paper.policy.seal({"schema": paper.SCHEMA, "backends": list(paper.BACKENDS),
            "orders": [list(x) for x in paper.ORDERS], "repetitions": 8, "cpu": 207,
            "minimum_seconds": 1., "timeout": 86400., "poles": {"uv": True, "ir": True},
            "normal_rescue": True, "acceptance_eligible": False,
            "cases": {cid: {} for cid in (*paper.CORE, *paper.LOOPS)},
            "sources": {name: paper.supplement.bound(ROOT / "tools" / name) for name in paper.SOURCES}})

    def test_hel_transformation_retains_timer_and_counters(self):
        original = (ROOT / "tools/ninjars_benchmark_worker.f").read_text()
        modified = paper.helicity_worker(original, 109)
        restored = modified.replace("      INTEGER NEXTERNAL,HELICITY\n      PARAMETER(HELICITY=109)\n", "      INTEGER NEXTERNAL\n").replace(
            "CALL @PREFIX@SLOOPMATRIXHEL_THRES(P,HELICITY,ANS,\n     $        -1D0,ACC,CODE)",
            "CALL @PREFIX@SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)")
        self.assertEqual(restored, original)
        self.assertEqual(modified.count("SLOOPMATRIXHEL_THRES"), 3)
        self.assertIn("FORCE_STABILITY_CHECK(.TRUE.)", modified)
        self.assertIn("TIMED_CODES(CODE)=TIMED_CODES(CODE)+1", modified)

    def test_refuse_changed_template_or_invalid_index(self):
        source = (ROOT / "tools/ninjars_benchmark_worker.f").read_text()
        for value in (0, -1, True, 1.2):
            with self.assertRaises(ValueError): paper.helicity_worker(source, value)
        with self.assertRaises(ValueError): paper.helicity_worker(source.replace("SLOOPMATRIX_THRES", "OTHER", 1), 1)

    def test_balanced_same_physical_backend_schedule(self):
        self.assertEqual(len(paper.ORDERS), 8)
        for position in range(4):
            self.assertEqual({b: sum(order[position] == b for order in paper.ORDERS) for b in paper.BACKENDS}, {1: 2, 6: 2, 7: 2, 8: 2})
        self.assertTrue(all(set(order) == set(paper.BACKENDS) for order in paper.ORDERS))

    def test_spec_rejects_scope_relaxation(self):
        self.assertEqual(paper.validate_spec(self.minimal_spec())["repetitions"], 8)
        for field, value in (("normal_rescue", False), ("acceptance_eligible", True), ("cpu", 190),
                             ("repetitions", 7), ("backends", [6, 8]), ("poles", {"uv": False, "ir": True})):
            spec = self.minimal_spec(); spec[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): paper.validate_spec(paper.policy.seal(spec))

    def test_missing_or_changed_producer_source_rejected(self):
        spec = self.minimal_spec(); spec["sources"].pop("ninjars_profile_v2.py")
        with self.assertRaises(ValueError): paper.validate_spec(paper.policy.seal(spec))
        spec = self.minimal_spec(); spec["sources"][paper.Path(paper.__file__).name]["sha256"] = "0" * 64
        with self.assertRaises(ValueError): paper.validate_spec(paper.policy.seal(spec))

    def test_selection_excludes_zero_negative_and_uses_physical_tuple(self):
        table = self.root / "table"; table.write_text("1 -1\n-1 1\n-1 -1\n0 0\n")
        weights = self.root / "weights"; weights.write_text("1 2 0 -10000\n")
        record = {"baseline_helicity_table": paper.supplement.bound(table), "baseline_current_filter": paper.supplement.bound(weights),
            "selected_one_based_index": 2, "selected_physical_tuple": [-1, 1]}
        self.assertEqual(paper.table_selection(record, table), (2, [-1, 1]))
        other = self.root / "different-table"; other.write_text("1 -1\n-1 -1\n-1 1\n0 0\n")
        with self.assertRaises(ValueError): paper.table_selection(record, other)

    def test_dp_and_qp_counts_use_actual_calls(self):
        for backend, code in ((1, 211), (6, 216), (7, 217), (8, 322)):
            self.raw(self.root, code, 3)
            row = paper.parse_observation(self.root, backend, 100, 3)
            self.assertTrue(row["physical_valid"])
            self.assertEqual(sum(row["timed_status_counts"].values()), 300)
            self.assertEqual(sum(row["status_counts"].values()), 100)

    def test_failed_collier_cost_is_retained_not_accepted(self):
        self.raw(self.root, 410, 3)
        row = paper.parse_observation(self.root, 7, 100, 3)
        self.assertFalse(row["physical_valid"])
        self.assertEqual(row["timed_status_counts"], {"410": 300})
        self.assertEqual(row["seconds"], 2.)
        self.assertEqual(row["seconds_per_point"], 2. / 300)

    def test_different_backend_and_missing_accuracy_are_failures(self):
        self.raw(self.root, 218)
        self.assertFalse(paper.parse_observation(self.root, 1, 100, 1)["physical_valid"])
        self.raw(self.root, 218, accuracy=-1.)
        self.assertFalse(paper.parse_observation(self.root, 8, 100, 1)["physical_valid"])

    def test_reject_wrong_histogram_missing_rows_and_nonfinite(self):
        self.raw(self.root)
        (self.root / "timed_status.dat").write_text("218 99\n")
        with self.assertRaises(ValueError): paper.parse_observation(self.root, 8, 100, 1)
        self.raw(self.root)
        path = self.root / "values.dat"; path.write_text(path.read_text().split("\n", 1)[1])
        with self.assertRaises(ValueError): paper.parse_observation(self.root, 8, 100, 1)
        self.raw(self.root, accuracy=float("nan"))
        with self.assertRaises(ValueError): paper.parse_observation(self.root, 8, 100, 1)

    def test_reject_split_order_status_disagreement(self):
        self.raw(self.root)
        path = self.root / "values.dat"; path.write_text(path.read_text().replace("1 1 218", "1 1 322", 1))
        with self.assertRaises(ValueError): paper.parse_observation(self.root, 8, 100, 1)

    def test_calibration_is_not_eighth_repetition(self):
        for values in ([1.] * 7, [1.] * 9, [1.] * 7 + [0.]):
            with self.assertRaises(ValueError): paper.distribution(values)
        self.assertEqual(paper.distribution([1.] * 8)["confidence_interval95"], [1., 1.])

    def test_explicit_handoff_before_any_build_or_engine(self):
        with mock.patch.object(paper, "preflight") as preflight:
            with self.assertRaises(ValueError): paper.collect(self.root / "none", self.root / "none", False)
            preflight.assert_not_called()
        path = self.root / "spec.json"; paper.bench.dump(path, self.minimal_spec())
        handoff = self.root / "handoff.json"; paper.bench.dump(handoff, {})
        with mock.patch.object(paper, "preflight") as preflight:
            with self.assertRaises(ValueError): paper.collect(path, handoff, True)
            preflight.assert_not_called()

    def test_private_build_rejects_extra_source_or_wrong_full_runtime(self):
        retained = self.root / "builds/case-candidate-8"; retained.mkdir(parents=True)
        source = paper.helicity_worker((ROOT / "tools/ninjars_benchmark_worker.f").read_text(), 2)
        source = source.replace("@PREFIX@", "ML5_0_").replace("@NEXTERNAL@", "3").replace("@NBORN@", "1")
        (retained / "worker.f").write_text(source)
        (retained / "param_card.dat").write_text("physical card")
        (retained / "MadLoopParams.dat").write_text("numerical card")
        files = {"SubProcesses/P0_case/check_sa.f": {"sha256": "original", "bytes": 1},
            "Source/other.f": {"sha256": "other", "bytes": 1}}
        original = self.root / "original.json"; paper.bench.dump(original, {"files": files})
        files = copy.deepcopy(files); files["SubProcesses/P0_case/check_sa.f"] = {
            "sha256": paper.workers.digest(retained / "worker.f"), "bytes": len(source)}
        compiled = {"files": files, "files_sha256": paper.bench.content_digest(files)}
        paper.bench.dump(retained / "compiled-sources.json", compiled)
        state = {"compiled_manifest": compiled, "runtime": {"full": "runtime"}, "libraries": {}, "runtime_metadata": {},
            "executable_sha256": "executable", "parameter_card_sha256": paper.workers.digest(retained / "param_card.dat"),
            "madloop_card_sha256": paper.workers.digest(retained / "MadLoopParams.dat")}
        build = {k: state[k] for k in ("executable_sha256", "parameter_card_sha256", "madloop_card_sha256")}
        build.update(driver_sha256=paper.workers.digest(retained / "worker.f"), generated_sources_sha256=compiled["files_sha256"])
        paper.bench.dump(retained / "build.json", build)
        case = {"definition": {"id": "case", "nfinal": 1}, "helicity_index": 2,
            "origin": {"compiled_manifest": paper.supplement.bound(original), "reducer_runtime": state["runtime"],
                "physical_card": paper.supplement.bound(retained / "param_card.dat")}}
        with mock.patch.object(paper.policy, "reducer_signature", return_value=state["runtime"]):
            self.assertEqual(paper.check_build(case, 8, self.root, state), build)
            wrong_card = copy.deepcopy(case); wrong_card["origin"]["physical_card"]["sha256"] = "other"
            with self.assertRaisesRegex(ValueError, "differs from origin"): paper.check_build(wrong_card, 8, self.root, state)
            wrong = {**state, "runtime": {"full": "different"}}
            with self.assertRaises(ValueError): paper.check_build(case, 8, self.root, wrong)
            changed = copy.deepcopy(compiled); changed["files"]["Source/other.f"]["sha256"] = "changed"
            paper.bench.dump(retained / "compiled-sources.json", changed)
            with self.assertRaisesRegex(ValueError, "more than the driver"): paper.check_build(case, 8, self.root, state)


class PaperRunnerTest(unittest.TestCase):
    setUp = PaperTimingTest.setUp
    raw = PaperTimingTest.raw

    def setup_runner(self, fail=False):
        self.report, self.work = self.root / "report", self.root / "work"
        (self.report / "points").mkdir(parents=True)
        (self.report / "points/case.dat").write_text("100\n")
        card = self.root / "physical"; card.write_text("physical")
        self.case = {"definition": {"id": "case"}, "origin": {"physical_card": paper.supplement.bound(card), "reducer_runtime": {}},
            "helicity_index": 2, "physical_helicity": [-1, 1]}
        self.commands, self.instances = [], []
        owner = self

        def prepare(case, backend, config, work, report):
            job = work / str(backend); directory = job / "outputs/candidate/case/SubProcesses/P0_case"
            directory.mkdir(parents=True)
            return job, directory, {"backend": backend, "parameter_card_sha256": "physics", "executable_sha256": str(backend),
                "shared_numerical_settings": {"collierrequiredaccuracy": 1e-12, "mlstabthres": 1e-9,
                    "colliercomputeuvpoles": True, "colliercomputeirpoles": True, **({"selected_backend_qp_integrand": True} if backend != 7 else {})}}

        class FakeWorker:
            def __init__(self, command, directory, log, **kwargs):
                self.directory = directory; self.backend = int(directory.parents[4].name)
                self.points, self.ndim, self.nso, self.nborn, self.sequence, self.closed = 100, 1, 1, 1, 0, False
                owner.instances.append(self)

            def run(self, loops):
                self.sequence += 1
                owner.commands.append((self.backend, self.sequence, loops))
                owner.raw(self.directory, {1: 211, 6: 216, 7: 410, 8: 322}[self.backend], loops)
                if fail and self.backend == 7 and self.sequence == 2: raise RuntimeError("controlled command failure")
                return .1

            def close(self): self.closed = True

        for obj, name, value in ((paper, "prepare_target", prepare), (paper, "target_state", mock.Mock(return_value={})),
                (paper.workers, "Worker", FakeWorker), (paper.workers, "validate_param_card", mock.Mock()),
                (paper.ownership, "profile_environment", mock.Mock(return_value={}))):
            patcher = mock.patch.object(obj, name, value); patcher.start(); self.addCleanup(patcher.stop)
        return {"relative_tolerance": 1e-7, "scale_tolerance": 1e-10}, {"cpu": 207, "timeout": 3., "minimum_seconds": 1., "orders": [list(x) for x in paper.ORDERS]}

    def test_failed_physics_keeps_eight_blocks_all_backends_same_counts(self):
        config, spec = self.setup_runner()
        paper.run_case(self.case, config, spec, self.work, self.report, paper.pair.Dispatch(self.report))
        expected = [(b, 1, 1) for b in paper.BACKENDS] + [(b, r + 2, 1) for r, order in enumerate(paper.ORDERS) for b in order]
        self.assertEqual(self.commands, expected)
        self.assertTrue(all(worker.closed for worker in self.instances))
        records = [paper.read(self.report / "data" / ("case-7-" + str(r) + ".json")) for r in range(8)]
        self.assertTrue(all(not row["physical_valid"] for row in records))
        self.assertEqual(sum(sum(row["timed_status_counts"].values()) for row in records), 800)

    def test_infrastructure_failure_retains_raw_and_closes_every_worker(self):
        config, spec = self.setup_runner(True)
        with self.assertRaisesRegex(RuntimeError, "controlled command failure"):
            paper.run_case(self.case, config, spec, self.work, self.report, paper.pair.Dispatch(self.report))
        self.assertTrue(all(worker.closed for worker in self.instances))
        self.assertTrue((self.report / "raw/case-candidate-7/measurement-0/values.dat").is_file())
        self.assertTrue((self.report / "failures/case.json").is_file())


class PaperReaderTest(unittest.TestCase):
    """A bounded synthetic raw run exercises the diagnostic reader contract."""
    setUp = PaperTimingTest.setUp
    raw = PaperTimingTest.raw
    def fixture(self, failed_collier=False):
        report = self.root / "report"; report.mkdir()
        self.report = report
        (report / "points").mkdir(); (report / "points/case.dat").write_text("100\nfixed\n")
        self.case = {"definition": {"id": "case"}, "points": paper.supplement.bound(report / "points/case.dat"),
            "helicity_index": 2, "physical_helicity": [-1, 1]}
        spec = {"cases": {"case": self.case}, "minimum_seconds": 1., "orders": [list(x) for x in paper.ORDERS],
            "candidate_cohort": {"runtime_identity_sha256": "runtime"}, "color_mode": "summed", "poles": {"uv": True, "ir": True},
            "sources": {}, "numerator_dp": {"status": "pending"}}
        paper.bench.dump(report / "input-spec.json", spec)
        specification = paper.supplement.bound(report / "input-spec.json")
        paper.bench.dump(report / "protocol.json", {"schema": paper.RESULT, "specification": specification,
            "thread_environment": {k: "1" for k in paper.pair.THREADS}, "normal_rescue": True, "acceptance_eligible": False,
            "configuration": {"relative_tolerance": 1e-7, "scale_tolerance": 1e-10}})
        paper.bench.dump(report / "run-status.json", {"status": "complete", "requested_cases": ["case"], "completed_cases": ["case"]})
        paper.bench.dump(report / "schedules/case.json", {"common_loops": 1, "orders": spec["orders"], "calibrations_are_measurements": False})
        paper.bench.dump(report / "states/case.json", {str(b): {} for b in paper.BACKENDS})
        self.builds = {}
        for b in paper.BACKENDS:
            self.builds[b] = {"parameter_card_sha256": "physics", "madloop_card_sha256": "settings", "executable_sha256": str(b),
                "driver_sha256": "driver", "generated_sources_sha256": "generated",
                "shared_numerical_settings": {"collierrequiredaccuracy": 1e-12, "mlstabthres": 1e-9,
                    "colliercomputeuvpoles": True, "colliercomputeirpoles": True, **({"selected_backend_qp_integrand": True} if b != 7 else {})}}
        dispatch = []
        sequence = [("calibration", "calibration", b) for b in paper.BACKENDS] + [("measurement", r, b) for r, order in enumerate(paper.ORDERS) for b in order]
        for phase, repetition, b in sequence:
            code = {1: 211, 6: 216, 7: 410 if failed_collier else 217, 8: 322}[b]
            directory = report / "raw" / ("case-candidate-" + str(b)) / (phase + "-" + str(repetition))
            self.raw(directory, code)
            raw = {name: paper.supplement.bound(directory / name) for name in (*paper.workers.OUTPUTS, "dimensions.dat")}
            observation = paper.parse_observation(directory, b, 100, 1)
            index = len(dispatch)
            dispatch.append({"index": index, "case": "case", "event": "start", "phase": phase, "backend": b,
                "repetition": repetition, "loops": 1, "worker_sequence": 1 if phase == "calibration" else repetition + 2})
            dispatch.append({**dispatch[-1], "index": index + 1, "event": "complete", "physical_valid": observation["physical_valid"]})
            row = {**self.builds[b], **observation, "case": "case", "source": "candidate", "backend": b, "repetition": repetition,
                "phase": phase, "loops": 1, "cpu": 207, "helicity_index": 2, "physical_helicity": [-1, 1], "acceptance_eligible": False,
                "points_sha256": self.case["points"]["sha256"], "dispatch_index": index, "raw_capture": raw,
                "worker_sequence": dispatch[-1]["worker_sequence"]}
            paper.bench.dump(report / "data" / ("case-" + str(b) + "-" + str(repetition) + ".json"), row)
        (report / "dispatch.jsonl").write_text("".join(json.dumps(x) + "\n" for x in dispatch))
        self.specification = specification
        self.reseal()
        self.stack = mock.patch.object(paper, "validate_spec", side_effect=lambda value: value)
        self.stack.start(); self.addCleanup(self.stack.stop)
        self.buildmock = mock.patch.object(paper, "check_build", side_effect=lambda case, b, report, state: self.builds[b])
        self.buildmock.start(); self.addCleanup(self.buildmock.stop)
        # Keep reader tests bounded; distribution itself is tested separately.
        self.distmock = mock.patch.object(paper, "distribution", side_effect=lambda values: {"median_seconds_per_point": sum(values)/8})
        self.distmock.start(); self.addCleanup(self.distmock.stop)
        return report / "result.json"

    def reseal(self):
        path = self.report / "result.json"
        files = {paper.policy.relative(p): paper.workers.digest(p) for p in self.report.rglob("*") if p.is_file() and p != path}
        paper.bench.dump(path, paper.policy.seal({"schema": paper.RESULT, "report": paper.policy.relative(self.report),
            "run_files_sha256": files, "status": "complete", "acceptance_eligible": False, "specification": self.specification}))

    def edit(self, backend, repetition, function):
        path = self.report / "data" / ("case-" + str(backend) + "-" + str(repetition) + ".json")
        row = paper.read(path); function(row); paper.bench.dump(path, row); self.reseal()

    def test_reader_failed_cell_counts_and_preserves_other_tools(self):
        result = self.fixture(True)
        summary = paper.report_snapshot(result)
        cell = summary["cases"]["case"]
        self.assertFalse(cell["all_four_physics_valid"])
        self.assertEqual(cell["backend"]["7"]["status"], "failed_physics")
        self.assertEqual(cell["backend"]["7"]["failed"], 800)
        self.assertEqual(cell["backend"]["8"]["qp"], 800)
        self.assertEqual(cell["backend"]["1"]["dp"], 800)
        self.assertEqual(cell["backend"]["7"]["timing_repetitions"], 8)
        self.assertEqual(len(cell["backend"]["7"]["raw_measurements"]), 8)

    def test_reader_rejects_calibration_promoted_as_raw_measurement(self):
        result = self.fixture()
        calibration = paper.read(self.report / "data/case-8-calibration.json")
        self.edit(8, 0, lambda r: r.update(raw_capture=calibration["raw_capture"]))
        with self.assertRaisesRegex(ValueError, "Calibration/reused"): paper.report_snapshot(result)

    def test_reader_rejects_changed_helicity_runtime_or_point_identity(self):
        for key, value in (("helicity_index", 3), ("executable_sha256", "other"), ("points_sha256", "other")):
            if hasattr(self, "report"):
                # Separate raw fixture directories without discarding prior proof.
                self.root = self.root / "next"; self.root.mkdir()
            result = self.fixture()
            self.edit(8, 0, lambda r: r.update({key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError): paper.report_snapshot(result)

    def test_reader_reparses_raw_not_recorded_pass_booleans(self):
        result = self.fixture(True)
        self.edit(7, 0, lambda r: r.update(physical_valid=True, physical_failures=[]))
        with self.assertRaises(ValueError): paper.report_snapshot(result)

    def test_reader_rejects_missing_eighth_block_or_changed_order(self):
        result = self.fixture()
        path = self.report / "data/case-8-7.json"; path.unlink(); self.reseal()
        with self.assertRaises((FileNotFoundError, ValueError)): paper.report_snapshot(result)

    def test_reader_rejects_calibration_only_and_undeclared_extra_attempt(self):
        result = self.fixture()
        paper.bench.dump(self.report / "data/undeclared-retry.json", {})
        self.reseal()
        with self.assertRaisesRegex(ValueError, "Undeclared"): paper.report_snapshot(result)


if __name__ == "__main__":
    unittest.main()
