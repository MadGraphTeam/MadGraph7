"""Full-stream diagnostic contracts, using synthetic data and recording stubs."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark_precision_stream as stream


def numerical_source():
    return """      MP_DONE=.FALSE.
      MP_DONE_ONCE=.FALSE.
      MP_PS_SET=.FALSE.
      STAB_INDEX=0
      DOING_QP_EVALS=.FALSE.
      IF(MP_DONE_ONCE) THEN
         RET_CODE_T=2
      ELSE
         RET_CODE_T=1
      ENDIF
      COMMON/P_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,
     $ FPE_IN_QP_REDUCTION
"""


class PrecisionStreamTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.spec = {"loops": 1, "chain": [9, 8, 6, 7, 1]}

    def fixture(self, codes=None):
        codes = codes or {}
        calls, values, final, reference = [], [], [], []
        for point in range(1, 101):
            code = codes.get(point, 219)
            start = (point - 1) * 5 + 1
            calls.append(f"1 {point} {code} {start} {start+2} 1 0 1 0")
            for order in range(2):
                nums = f"{point}.0 {point+order}.0 0.25 0.125 1e-9"
                values.append(f"1 {point} {order} {code} {nums}")
                final.append(f"{point} {order} {code} {nums}")
                reference.append({"point": point, "order": order, "status": 219,
                                  "coefficients": [float(point), float(point + order), .25, .125],
                                  "accuracy": 1e-9})
        self.write("stream-calls.dat", calls)
        self.write("stream-values.dat", values)
        self.write("values.dat", final)
        self.write("dimensions.dat", ["1 1 1"])
        self.write("stream-clock.dat", ["1000000000 9223372036854775807 0 502"])
        self.write("timing.dat", ["1.25 0.000000502"])
        from collections import Counter
        histogram = Counter(codes.get(p, 219) for p in range(1, 101))
        self.write("timed_status.dat", [f"{code} {count}" for code, count in histogram.items() if 0 <= code <= 499])
        self.control("clock-before.dat")
        self.control("clock-after.dat")
        return {"rows": reference}

    def write(self, name, lines):
        (self.directory / name).write_text("\n".join(lines) + "\n")

    def control(self, name, rate=1000000000):
        self.write(name, [f"{rate} 9223372036854775807 0 30002 10000",
                          *(f"{3*i+1} {3*i+2}" for i in range(10000))])

    def parse(self, reference):
        return stream.parse_batch(self.directory, self.spec, reference)

    def test_aliases_t_flags_and_fallback_are_distinct(self):
        aliases = ((219, 229, 323, 9), (218, 228, 322, 8),
                   (216, 226, 328, 6), (211, 221, 329, 1))
        for dp, mixed, qp, backend in aliases:
            with self.subTest(backend=backend):
                self.assertEqual(stream.classify(dp, self.spec["chain"], True, True)["category"], "accepted_dp_no_observed_qp")
                self.assertEqual(stream.classify(mixed, self.spec["chain"], True, True)["category"], "accepted_dp_after_qp")
                actual = stream.classify(qp, self.spec["chain"], True, True)
                self.assertEqual(actual["category"], "accepted_qp")
                self.assertEqual(actual["actual_backend"], backend)
                self.assertEqual(actual["fallback_backend"], backend != 9)
                contradictory = qp - 10
                self.assertEqual(stream.classify(contradictory, self.spec["chain"], True, True)["category"], "failed_or_inconsistent")
        self.assertEqual(stream.classify(227, [7], True, True)["category"], "accepted_dp_after_qp")
        self.assertEqual(stream.classify(120, [9], True, True)["category"], "unchecked")
        self.assertIsNone(stream.classify(120, [9], True, True)["actual_backend"])
        self.assertEqual(stream.classify(139, [9], True, True)["category"], "failed_or_inconsistent")
        self.assertEqual(stream.classify(419, [9], True, True)["category"], "failed_or_inconsistent")
        self.assertEqual(stream.classify(219, [9], False, True)["category"], "failed_nonfinite")

    def test_complete_stream_retains_mixed_unchecked_failed_samples(self):
        reference = self.fixture({2: 229, 3: 323, 4: 120, 5: 419})
        result = self.parse(reference)
        self.assertEqual(len(result["calls"]), 100)
        self.assertEqual(result["groups"]["accepted_dp_after_qp"]["calls"], 1)
        self.assertEqual(result["groups"]["accepted_qp"]["calls"], 1)
        self.assertEqual(result["groups"]["unchecked"]["calls"], 1)
        self.assertEqual(result["groups"]["failed_or_inconsistent"]["calls"], 1)
        self.assertFalse(result["all_calls_checked"])
        self.assertTrue(result["coefficient_comparison"]["passed"])
        self.assertEqual(len(result["status_accuracy_differences"]), 8)
        # A sticky software latch is retained, never treated as a new failure.
        self.assertEqual(result["calls"][0]["software_fpe_before"], [1, 0])
        self.assertEqual(result["calls"][0]["category"], "accepted_dp_no_observed_qp")
        self.assertFalse(result["performance_acceptance_eligible"])
        self.assertFalse(result["clock_overhead_subtracted"])

    def test_failed_nonfinite_values_survive_without_json_nan(self):
        reference = self.fixture()
        for name in ("stream-values.dat", "values.dat"):
            path = self.directory / name
            path.write_text(path.read_text().replace("1.0 1.0 0.25", "1.0 NaN 0.25", 1))
        result = self.parse(reference)
        self.assertEqual(result["calls"][0]["category"], "failed_nonfinite")
        self.assertIsNone(result["final_rows"][0]["coefficients"][1])
        json.dumps(result, allow_nan=False)

    def test_omitted_reordered_or_subset_calls_rejected(self):
        reference = self.fixture()
        original = (self.directory / "stream-calls.dat").read_text().splitlines()
        for lines in (original[:-1], [original[1], original[0], *original[2:]], original[::2]):
            self.write("stream-calls.dat", lines)
            with self.assertRaisesRegex(ValueError, "full-stream calls"):
                self.parse(reference)

    def test_clock_wrap_zero_and_invalid_inputs(self):
        self.assertEqual(stream.tick_delta(98, 1, 99), 3)
        self.assertEqual(stream.tick_delta(0, 0, 2**63 - 1), 0)
        for triple in ((-1, 2, 99), (2, 100, 99), (1, 2, 0), (True, 2, 99)):
            with self.subTest(triple=triple), self.assertRaises(ValueError):
                stream.tick_delta(*triple)
        reference = self.fixture()
        path = self.directory / "stream-calls.dat"
        path.write_text(path.read_text().replace("219 1 3 ", "219 1 1 ", 1))
        result = self.parse(reference)
        self.assertEqual(result["groups"]["accepted_dp_no_observed_qp"]["zero_tick_calls"], 1)

    def test_clock_controls_and_outer_chronology_reconciled(self):
        reference = self.fixture()
        clock = {"clock_maximum": 9999, "clock_rate": 100, "outer_seconds": 1.0}
        stream.validate_controller(1.5, clock)
        for seconds in (.5, 100.0, float("nan")):
            with self.subTest(seconds=seconds), self.assertRaisesRegex(ValueError, "Controller interval"):
                stream.validate_controller(seconds, clock)
        self.control("clock-after.dat", rate=2)
        with self.assertRaisesRegex(ValueError, "rate/max changed"):
            self.parse(reference)
        self.control("clock-after.dat")
        path = self.directory / "stream-calls.dat"
        path.write_text(path.read_text().replace("219 6 8 ", "219 1 8 ", 1))
        with self.assertRaisesRegex(ValueError, "chronology"):
            self.parse(reference)

    def test_histogram_timing_and_final_rows_cannot_disagree(self):
        reference = self.fixture()
        for name, changed, message in (("timed_status.dat", "219 99\n", "histogram"),
                                       ("timing.dat", "1.25 1.0\n", "raw outer ticks"),
                                       ("values.dat", "1 0 219 1 2 3 4 1e-9\n", "final traversal")):
            path = self.directory / name
            before = path.read_text()
            path.write_text(changed)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                self.parse(reference)
            path.write_text(before)

    def test_final_coefficient_mismatch_is_retained_open_result(self):
        reference = self.fixture()
        for name in ("stream-values.dat", "values.dat"):
            path = self.directory / name
            path.write_text(path.read_text().replace("1.0 1.0 0.25", "1.0 9.0 0.25", 1))
        result = self.parse(reference)
        self.assertFalse(result["coefficient_comparison"]["passed"])
        self.assertEqual(len(result["calls"]), 100)

    def test_complete_sealed_prerequisite_is_required(self):
        (self.directory / "report").mkdir()
        value = {"cpu": 12}
        with patch.object(stream.bench, "REPORT", self.directory / "report"), patch.object(stream.local, "verify_numerical", return_value={"case": ["proof"]}) as replay:
            with self.assertRaisesRegex(ValueError, "seal"):
                stream.sealed_prerequisite(value)
            seal = self.directory / "numerical-prerequisites.json"
            seal.write_text('{"case": []}')
            with self.assertRaisesRegex(ValueError, "seal"):
                stream.sealed_prerequisite(value)
            seal.write_text('{"case": ["proof"]}')
            self.assertEqual(stream.sealed_prerequisite(value), stream.bench.digest(seal))
            self.assertEqual(replay.call_count, 3)

    def test_prepared_specification_cannot_change_schedule_or_chain(self):
        report = self.directory / "diagnostic"
        report.mkdir()
        workspace = self.directory / "work"
        output = workspace / "outputs/candidate/case-a"
        output.mkdir(parents=True)
        (self.directory / "measurement.json").write_text("{}")
        (report / "points.dat").write_text("100\n")
        (report / "reference.json").write_text("{}")
        spec = {"schema": stream.SCHEMA, **stream.SCOPE, "cpu": 12,
                "measurement": "m", "label": "l", "source": "candidate",
                "case": {"id": "case-a"}, "chain": [9], "loops": 1,
                "local_manifest_sha256": stream.bench.digest(self.directory / "measurement.json"),
                "numerical_prerequisite_sha256": "seal", "protocols": {}}
        (report / "specification.json").write_text(json.dumps(spec))
        prep = {"original": str(self.directory / "original"), "original_before": {"objects": {}},
                "output": str(output), "resources": {},
                "specification_sha256": stream.bench.digest(report / "specification.json"),
                "points_sha256": stream.bench.digest(report / "points.dat"),
                "reference_sha256": stream.bench.digest(report / "reference.json")}
        (report / "preparation.json").write_text(json.dumps(prep))
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.object(stream, "PROTOCOLS", ()))
            stack.enter_context(patch.object(stream, "roots", return_value=(report, workspace)))
            stack.enter_context(patch.object(stream.local, "roots", return_value=(self.directory, workspace)))
            stack.enter_context(patch.object(stream, "sealed_prerequisite", return_value="seal"))
            stack.enter_context(patch.object(stream, "immutable_inventory", return_value={"objects": {}}))
            native = stack.enter_context(patch.object(stream.subprocess, "run", side_effect=AssertionError("No native execution permitted")))
            for change in ({"loops": 2}, {"chain": [8, 9]}):
                altered = dict(spec, **change)
                (report / "specification.json").write_text(json.dumps(altered))
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "retained inputs"):
                    stream.verify_preparation(report, altered, {"cpu": 12}, compiled=True)
            native.assert_not_called()

    def test_built_guard_rejects_card_and_numerical_object_changes(self):
        output = self.directory / "output"
        directory = output / "SubProcesses/P0_test"
        directory.mkdir(parents=True)
        (directory / "check").write_text("executable")
        receipt = {"sources": {"f": "source"}, "objects": {"o": "object"},
                   "executable_sha256": stream.bench.digest(directory / "check"),
                   "dependencies": {}, "cards": {"param_card.dat": "physical"},
                   "runtime": {"libraries": {}, "archives": {}, "metadata": {}}, "compiler_flags": {}}
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.object(stream.bench, "runtime_identity", return_value=receipt["runtime"]))
            stack.enter_context(patch.object(stream.bench, "generated_sources", return_value=receipt["sources"]))
            objects = stack.enter_context(patch.object(stream, "object_inventory", return_value=receipt["objects"]))
            stack.enter_context(patch.object(stream.bench, "dependency_inputs", return_value={}))
            cards = stack.enter_context(patch.object(stream, "card_inventory", return_value=receipt["cards"]))
            stack.enter_context(patch.object(stream, "capture_effective_flags", return_value={}))
            stack.enter_context(patch.object(stream.bench, "same_compiler_policy", return_value=True))
            stream.check_built(output, receipt)
            cards.return_value = {"param_card.dat": "different"}
            with self.assertRaisesRegex(ValueError, "build/runtime changed"):
                stream.check_built(output, receipt)
            cards.return_value = receipt["cards"]
            objects.return_value = {"o": "different"}
            with self.assertRaisesRegex(ValueError, "build/runtime changed"):
                stream.check_built(output, receipt)

    def test_driver_link_guard_refuses_numerical_recipe(self):
        completed = subprocess.CompletedProcess([], 0, stdout="gfortran -c loop_matrix.f -o loop_matrix.o\ngfortran -o check check_sa.o loop_matrix.o\n", stderr="")
        with patch.object(stream.cohorts.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(ValueError, "numerical-object build"):
                stream.cohorts.link_commands(self.directory, self.directory, {})

    def test_closed_receipt_cannot_hide_missing_or_changed_batch(self):
        rows = [{"directory": "batch-0000", "status": "complete"}]
        with self.assertRaisesRegex(ValueError, "all registered batches"):
            stream.validate_closed_batches(self.directory, {"batches": 2}, rows, {"batches": []})
        batch = self.directory / "batch-0000"
        batch.mkdir()
        (batch / "result.json").write_text("{}")
        result = {"batches": [{"directory": "batch-0000", "result_sha256": stream.bench.digest(batch / "result.json")}]}
        stream.validate_closed_batches(self.directory, {"batches": 1}, rows, result)
        (batch / "result.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            stream.validate_closed_batches(self.directory, {"batches": 1}, rows, result)

    def test_software_flag_unavailability_cannot_be_fabricated(self):
        reference = self.fixture()
        with self.assertRaisesRegex(ValueError, "FPE observations"):
            stream.parse_batch(self.directory, self.spec, reference, fpe_observed=False)
        path = self.directory / "stream-calls.dat"
        path.write_text(path.read_text().replace(" 1 0 1 0", " -1 0 1 0", 1))
        with self.assertRaisesRegex(ValueError, "FPE observations"):
            self.parse(reference)

    def test_refuse_relabelled_repeat_and_execution_lock_conflict(self):
        old = self.directory / "older"
        old.mkdir()
        (old / "specification.json").write_text(json.dumps({"case": {"id": "case-a"}, "lane": "candidate-9"}))
        with self.assertRaisesRegex(ValueError, "new label"):
            stream.refuse_duplicate(self.directory / "new", {"id": "case-a"}, "candidate-9")
        with patch.object(stream.local, "roots", return_value=(self.directory, self.directory)):
            with stream.measurement_lock("m"):
                with self.assertRaises(BlockingIOError):
                    with stream.measurement_lock("m"):
                        self.fail("Concurrent forward/diagnostic lock admitted")

    def test_clone_writable_symlink_escape_rejected(self):
        output = self.directory / "output"
        directory = output / "SubProcesses/P0_test"
        directory.mkdir(parents=True)
        target = self.directory / "original.f"
        target.write_text("original")
        (directory / "check_sa.f").symlink_to(target)
        with self.assertRaisesRegex(ValueError, "escapes"):
            stream.writable_targets(output)
        self.assertEqual(target.read_text(), "original")

    def test_call_local_t_semantics_guard_and_optional_fpe_render(self):
        source = numerical_source()
        stream.validate_t_semantics(source)
        for old, new in (("MP_DONE_ONCE=.FALSE.", "MP_DONE_ONCE=.TRUE."),
                         ("RET_CODE_T=1", "RET_CODE_T=2")):
            with self.assertRaisesRegex(ValueError, "T semantics"):
                stream.validate_t_semantics(source.replace(old, new))
        params = {"prefix": "P_", "nexternal": 3, "nborn": 1}
        driver, observed = stream.render(stream.TEMPLATE.read_text(), params, source)
        self.assertTrue(observed)
        self.assertNotIn("@", driver)
        self.assertIn("CALL_FPE(1:2,NCALL)", driver)
        self.assertNotIn("FPE_IN_DP_REDUCTION=.FALSE.", driver.upper())
        _, observed = stream.render(stream.TEMPLATE.read_text(), params, source.split("      COMMON/")[0])
        self.assertFalse(observed)

    def test_cli_help_execute_opt_in_and_bounds(self):
        completed = subprocess.run([sys.executable, str(ROOT / "tools/tirust_benchmark_precision_stream.py"), "--help"], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertIn("prepare", completed.stdout)
        self.assertIn("never speed acceptance", completed.stdout)
        with self.assertRaises(SystemExit):
            stream.parser().parse_args(["run", "--measurement", "m", "--label", "l"])
        for option, value in (("--loops", "0"), ("--loops", "10001"), ("--batches", "0"), ("--timeout", "nan")):
            with self.subTest(option=option, value=value), self.assertRaisesRegex(ValueError, "loops"):
                stream.main(["prepare", "--measurement", "m", "--label", "l", "--case", "c", "--lane", "candidate-9", option, value])


class SyntheticFortranRecordingTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("gfortran"), "GNU Fortran required for recording-only driver qualification")
    def test_real_driver_with_recording_stub_preserves_full_stream(self):
        # No MadLoop numerical objects, reducer or real process is linked.
        with tempfile.TemporaryDirectory(prefix="tirust-precision-recording-") as temporary:
            directory = Path(temporary)
            driver, _ = stream.render(stream.TEMPLATE.read_text(), {"prefix": "P_", "nexternal": 3, "nborn": 1}, numerical_source())
            (directory / "driver.f").write_text(driver)
            (directory / "coupl.inc").write_text("      REAL*8 MU_R\n      COMMON/TEST_MU/MU_R\n")
            (directory / "pmass.inc").write_text("      PMASS=0D0\n")
            (directory / "stubs.f90").write_text('''
subroutine setpara(name)
  character*(*) name
end
subroutine update_as_param()
end
subroutine P_get_answer_dimension(n)
  integer n
  n=1
end
subroutine P_get_nsqso_loop(n)
  integer n
  n=1
end
subroutine P_force_stability_check(active)
  logical active
end
subroutine P_sloopmatrix_thres(p,ans,threshold,acc,code)
  real*8 p(0:3,3),ans(0:3,0:1),threshold,acc(0:1)
  integer code,j,point,calls
  logical dp,qp
  common/P_FPE_IN_REDUCTION/dp,qp
  save calls
  data calls/0/
  calls=calls+1
  point=nint(p(0,3))
  code=219
  if(point==2) code=229
  if(point==3) code=323
  if(point==4) code=120
  ! During warmup accept all points; the measured call counter starts at 102.
  if(calls<=101) code=219
  dp=.true.
  qp=.false.
  do j=0,1
    ans(:,j)=(/dble(point),dble(point+j),0.25d0,0.125d0/)
  enddo
  acc=1d-9
end
subroutine P_exit_madloop()
end
''')
            command = [shutil.which("gfortran"), "-ffixed-line-length-none", "-fcheck=all", "driver.f", "stubs.f90", "-o", "check"]
            completed = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            points = directory / "points.dat"
            points.write_text("100\n" + "".join(f"10 0 0 10\n10 0 0 -10\n{p} 0 0 0\n" for p in range(1, 101)))
            with stream.Worker(["./check", str(points)], directory, directory / "worker.log", timeout=30) as worker:
                worker.run(2)
            calls = (directory / "stream-calls.dat").read_text().splitlines()
            self.assertEqual([tuple(map(int, line.split()[:2])) for line in calls], [(i, p) for i in (1, 2) for p in range(1, 101)])
            self.assertEqual(len((directory / "stream-values.dat").read_text().splitlines()), 400)
            # Every return, including unchecked, survives in the buffer.
            self.assertIn(120, [int(line.split()[2]) for line in calls])
            self.assertIn(229, [int(line.split()[2]) for line in calls])
            stream.clock_pairs(directory / "clock-before.dat")
            stream.clock_pairs(directory / "clock-after.dat")
            reference = {"rows": [{"point": point, "order": order, "status": 219,
                                    "coefficients": [float(point), float(point + order), .25, .125],
                                    "accuracy": 1e-9}
                                   for point in range(1, 101) for order in range(2)]}
            observation = stream.parse_batch(directory, {"loops": 2, "chain": [9]}, reference, fpe_observed=True)
            self.assertEqual(observation["groups"]["accepted_dp_after_qp"]["calls"], 2)
            self.assertEqual(observation["groups"]["accepted_qp"]["calls"], 2)
            self.assertEqual(observation["groups"]["unchecked"]["calls"], 2)
            self.assertFalse(observation["all_calls_checked"])
            self.assertTrue(observation["coefficient_comparison"]["passed"])


if __name__ == "__main__":
    unittest.main()
