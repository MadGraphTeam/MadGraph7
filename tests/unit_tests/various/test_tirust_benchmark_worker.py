"""The benchmark protocol must not turn failures or rescue aliases into passes."""
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "tirust_benchmark_worker", ROOT / "tools/tirust_benchmark_worker.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class TestParsing(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        (self.directory / "dimensions.dat").write_text("2 1 2\n")
        (self.directory / "timing.dat").write_text("0.1 0.2\n")
        (self.directory / "values.dat").write_text("".join(
            f"{point} {order} {status} 1 2 -3 -4 1e-10\n"
            for point, status in ((1, 219), (2, 323)) for order in range(3)))
        (self.directory / "timed_status.dat").write_text("219 2\n323 2\n")

    def test_every_coefficient_born_column_and_precision_is_retained(self):
        result = worker.parse_outputs(self.directory, 9, 2, 2)
        self.assertEqual(len(result["rows"]), 6)
        self.assertEqual(result["rows"][-1]["coefficients"], [1., 2., -3., -4.])
        self.assertEqual(result["backend_precision_counts"], {"9:DP": 2, "9:QP": 2})
        self.assertFalse(result["native_dp_only"])
        self.assertEqual(result["seconds_per_point"], .05)

    def test_aliases_use_both_hundreds_and_units(self):
        for code, backend, precision in ((219, 9, "DP"), (319, 1, "QP"),
                                         (313, 9, "QP"), (318, 6, "QP"),
                                         (218, 8, "DP"), (312, 8, "QP"),
                                         (211, 1, "DP"), (216, 6, "DP"),
                                         (217, 7, "DP")):
            with self.subTest(code=code):
                self.assertEqual(worker.validate_status(code, backend)["precision"], precision)
        for code in (209, 239, 119, 419, 213):
            with self.subTest(code=code), self.assertRaises(ValueError):
                worker.validate_status(code, 9)
        with self.assertRaises(ValueError):
            worker.validate_status(319, 9)

    def test_rescue_chain_preserves_actual_backend(self):
        path = self.directory / "values.dat"
        path.write_text(path.read_text().replace("323", "316"))
        # H=3/U=6 is invalid even when Ninja is selected.
        with self.assertRaises(worker.MeasurementError):
            worker.parse_outputs(self.directory, [9, 6], 2, 2)
        path.write_text(path.read_text().replace("316", "216"))
        (self.directory / "timed_status.dat").write_text("219 2\n216 2\n")
        result = worker.parse_outputs(self.directory, [9, 6], 2, 2)
        self.assertEqual(result["backend_precision_counts"], {"9:DP": 2, "6:DP": 2})
        self.assertFalse(result["native_dp_only"])

    def test_rejected_measurement_contains_unchanged_evidence(self):
        path = self.directory / "values.dat"
        raw = path.read_text().replace("219", "209")
        path.write_text(raw)
        with self.assertRaises(worker.MeasurementError) as caught:
            worker.parse_outputs(self.directory, 9, 2, 2)
        self.assertEqual(caught.exception.raw_outputs["values.dat"], raw)
        self.assertEqual(path.read_text(), raw)

    def test_missing_born_column_and_mismatched_histogram_fail(self):
        path = self.directory / "values.dat"
        original = path.read_text()
        path.write_text("\n".join(original.splitlines()[:-1]))
        with self.assertRaisesRegex(worker.MeasurementError, "Missing"):
            worker.parse_outputs(self.directory, 9, 2, 2)
        path.write_text(original)
        (self.directory / "timed_status.dat").write_text("219 4\n")
        with self.assertRaisesRegex(worker.MeasurementError, "absent"):
            worker.parse_outputs(self.directory, 9, 2, 2)

    def test_unstable_intermediate_timed_call_is_rejected(self):
        (self.directory / "timed_status.dat").write_text("219 1\n323 2\n210 1\n")
        with self.assertRaisesRegex(worker.MeasurementError, "Unexpected status"):
            worker.parse_outputs(self.directory, 9, 2, 2)


class TestPairedStatistics(unittest.TestCase):
    def test_constant_ratio_and_minimum_pairs(self):
        result = worker.paired_statistics([1.]*7, [.9]*7)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["upper_95"], .9)
        self.assertFalse(worker.paired_statistics([1.]*6, [.9]*6)["passed"])
        self.assertFalse(worker.paired_statistics([1.]*7, [1.06]*7)["passed"])

    def test_uncertainty_can_fail_a_faster_mean(self):
        result = worker.paired_statistics([1.]*7, [.7, 1.3, .7, 1.3, .7, 1.3, .7])
        self.assertLess(result["geometric_mean_ratio"], 1.)
        self.assertGreater(result["upper_95"], 1.05)
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["student_t_critical"], 2.446912)

    def test_pairing_changes_uncertainty(self):
        timings = [1., 2., 3., 4., 5., 6., 7.]
        paired = worker.paired_statistics(timings, [x*.95 for x in timings])
        shuffled = worker.paired_statistics(timings, list(reversed(timings)))
        self.assertTrue(paired["passed"])
        self.assertFalse(shuffled["passed"])

    def test_invalid_data_fail_closed(self):
        for baseline, candidate in (([1], [1]), ([1, 2], [1]),
                                    ([1, 0], [1, 1]), ([1, math.nan], [1, 1])):
            with self.subTest(baseline=baseline), self.assertRaises(ValueError):
                worker.paired_statistics(baseline, candidate)


class TestController(unittest.TestCase):
    def test_constructor_protocol_failure_preserves_new_and_previous_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "timing.dat").write_text("previous measurement\n")
            script = directory / "fake.py"
            script.write_text('''import pathlib, sys
fifo = open(sys.argv[-1], 'w', buffering=1)
pathlib.Path('warmup.dat').write_text('current startup evidence\\n')
print('INVALID READY', file=fifo)
sys.stdin.read()
''')
            with self.assertRaises(worker.ProtocolError) as caught:
                worker.Worker([sys.executable, script], directory,
                              directory / "worker.log", timeout=5.)
            error = caught.exception
            self.assertEqual(error.raw_outputs, {"warmup.dat": "current startup evidence\n"})
            self.assertEqual((error.raw_directory / "warmup.dat").read_text(),
                             "current startup evidence\n")
            self.assertEqual((error.raw_directory.parent / "previous-outputs/timing.dat").read_text(),
                             "previous measurement\n")
            metadata = json.loads((error.raw_directory / "startup.json").read_text())
            self.assertEqual(metadata["error"]["type"], "ProtocolError")
            self.assertFalse((directory / "worker-protocol.fifo").exists())

    def test_protocol_failure_archives_evidence_and_reaps_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            script = directory / "fake.py"
            script.write_text('''import pathlib, sys
fifo = open(sys.argv[-1], 'w', buffering=1)
print('diagnostics' * 100000, flush=True)
print('READY 1 1 0 0 0 .1', file=fifo)
command = sys.stdin.readline()
pathlib.Path('values.dat').write_text('raw failing result\\n')
print('ERROR synthetic-failure', file=fifo)
sys.stdin.read()
''')
            child = worker.Worker([sys.executable, script], directory,
                                  directory / "worker.log", timeout=5.)
            with self.assertRaises(worker.ProtocolError):
                child.run(1)
            self.assertIsNotNone(child.process.poll())
            self.assertFalse(child.pipe.exists())
            self.assertEqual((child.last_raw_directory / "values.dat").read_text(),
                             "raw failing result\n")
            metadata = json.loads((child.last_raw_directory / "command.json").read_text())
            self.assertEqual(metadata["error"]["type"], "ProtocolError")
            self.assertGreater((directory / "worker.log").stat().st_size, 1000000)


@unittest.skipUnless(shutil.which("gfortran"), "GNU Fortran is required")
class TestCompiledWorker(unittest.TestCase):
    def test_masses_full_sums_multiple_commands_and_born_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = (ROOT / "tools/tirust_benchmark_worker.f").read_text()
            source = source.replace("@NEXTERNAL@", "3").replace("@NBORN@", "2")
            source = source.replace("@PREFIX@", "STUB_")
            (directory / "driver.f").write_text(source)
            (directory / "coupl.inc").write_text("      REAL*8 MU_R\n      COMMON /SCALE/ MU_R\n")
            (directory / "pmass.inc").write_text("      PMASS=(/0D0,0D0,2D0/)\n")
            (directory / "stub.f90").write_text('''
subroutine SETPARA(card)
 character(*) card
 real*8 mu_r
 common /scale/ mu_r
 mu_r=2d0
end subroutine
subroutine UPDATE_AS_PARAM()
end subroutine
subroutine STUB_GET_ANSWER_DIMENSION(n)
 integer n
 n=2
end subroutine
subroutine STUB_GET_NSQSO_LOOP(n)
 integer n
 n=1
end subroutine
subroutine STUB_FORCE_STABILITY_CHECK(force)
 logical force
 if (.not.force) stop 91
end subroutine
subroutine STUB_SLOOPMATRIX_THRES(p,ans,threshold,acc,code)
 real*8 p(0:3,3),ans(0:3,0:2),threshold,acc(0:1)
 integer code,j,warm_status,warm_repeats
 character*32 configured_status
 integer, save :: calls=0
 calls=calls+1
 if (threshold.ne.-1d0) stop 92
 ans(:,0)=(/11d0,12d0,13d0,14d0/)
 ans(:,1)=(/21d0,22d0,23d0,24d0/)
 ans(0,2)=31d0
 acc=1d-10
 code=219
 if (p(3,1).lt.0d0) code=323
 call get_environment_variable('BENCHMARK_STUB_WARM_STATUS',configured_status)
 if (len_trim(configured_status).gt.0) then
   read(configured_status,*) warm_status
   warm_repeats=2
   if(calls.gt.1.and.calls.le.1+warm_repeats) code=warm_status
 endif
 if(calls.eq.1) code=109
end subroutine
subroutine STUB_EXIT_MADLOOP()
end subroutine
''')
            subprocess.run([shutil.which("gfortran"), "-ffixed-line-length-none",
                            "-fcheck=all", "-o", "check", "driver.f", "stub.f90"],
                           cwd=directory, check=True, capture_output=True)
            masses = subprocess.check_output([str(directory / "check"), "--masses"],
                                             cwd=directory, text=True)
            self.assertIn("MASSES", masses)
            self.assertEqual(float(masses.splitlines()[-1].split()[1]), 2.)
            self.assertEqual(list(map(float, (directory / "masses.dat").read_text().split())),
                             [0., 0., 2.])
            points = directory / "points.dat"
            points.write_text("2\n1 0 0 1\n1 0 0 -1\n2 0 0 0\n"
                              "1 0 0 -1\n1 0 0 1\n2 0 0 0\n")
            for restart in range(2):
                with worker.Worker([str(directory / "check"), points], directory,
                                   directory / f"worker-{restart}.log", timeout=5.) as child:
                    for loops in (1, 3):
                        child.run(loops)
                        result = worker.parse_outputs(directory, 9, 2, loops)
                        self.assertEqual(len(result["rows"]), 6)
                        self.assertEqual(result["rows"][0]["coefficients"], [11.,12.,13.,14.])
                        self.assertEqual(result["rows"][2]["coefficients"], [31.,0.,0.,0.])
                        self.assertEqual(result["backend_precision_counts"],
                                         {"9:DP": loops, "9:QP": loops})
                        self.assertEqual(worker.raw_outputs(child.last_raw_directory),
                                         worker.raw_outputs(directory))
                self.assertEqual(child.process.returncode, 0)
            for status in (139, 239):
                with self.subTest(initialization_status=status):
                    env = dict(os.environ, BENCHMARK_STUB_WARM_STATUS=str(status))
                    with worker.Worker([str(directory / "check"), points], directory,
                                       directory / f"warm-{status}.log", env=env, timeout=5.) as child:
                        trace = self.trace_rows(directory)
                        self.assertEqual([int(row[3]) for row in trace[::3]],
                                         [109, status, status, 219, 323])
                        self.assertEqual([int(row[1]) for row in trace[::3]], [1, 1, 2, 1, 2])
                        self.assertEqual(worker.raw_outputs(child.last_raw_directory),
                                         worker.raw_outputs(directory))
                        child.run(1)
                        self.assertEqual(worker.parse_outputs(directory, 9, 2, 1)["status_counts"],
                                         {"219": 1, "323": 1})
            for status in (249, 230, 130, 410, 213, 109):
                with self.subTest(failing_warmup_status=status):
                    env = dict(os.environ, BENCHMARK_STUB_WARM_STATUS=str(status))
                    with self.assertRaises(worker.MeasurementError) as caught:
                        worker.Worker([str(directory / "check"), points], directory,
                                      directory / f"warm-{status}.log", env=env, timeout=5.)
                    error = caught.exception
                    self.assertNotIsInstance(error, worker.ProtocolError)
                    self.assertIn(f"point 1, status {status}", str(error))
                    trace = self.trace_rows(directory)
                    self.assertEqual(len(trace), 6)
                    self.assertEqual([int(row[3]) for row in trace[::3]], [109, status])
                    self.assertEqual([float(value) for value in trace[-3][4:]],
                                     [11., 12., 13., 14., 1e-10])
                    self.assertEqual([float(value) for value in trace[-1][4:]],
                                     [31., 0., 0., 0., 1e-10])
                    self.assertEqual((error.raw_directory / "warmup.dat").read_text(),
                                     error.raw_outputs["warmup.dat"])
                    metadata = json.loads((error.raw_directory / "startup.json").read_text())
                    self.assertEqual(metadata["error"]["type"], "MeasurementError")
                    self.assertFalse((directory / "worker-protocol.fifo").exists())

    @staticmethod
    def trace_rows(directory):
        return [line.split() for line in (directory / "warmup.dat").read_text().splitlines()
                if not line.startswith("#")]


if __name__ == "__main__":
    unittest.main()
