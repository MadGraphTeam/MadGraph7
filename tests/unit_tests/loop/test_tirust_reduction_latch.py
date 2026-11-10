"""Compile latch lifecycle stubs against the LIVE optimized MadLoop template.

This reuses the sealed synthetic fixture. It tests software-latch control flow,
not complete MadLoop initialization, physical integrals, accuracy or timing.
"""
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

from madgraph import MG5DIR

ROOT = Path(MG5DIR)
LIVE = ROOT / "madgraph/iolibs/template_files/loop_optimized/loop_matrix_standalone.inc"
FIXTURE = ROOT / ("docs/benchmarkfolder/experiments/reduction-latch-lifecycle/"
                  "source-proposal1/stub_check.py")
FIXTURE_SHA256 = "c973c9b316d9bfd28cdaa5c405f8a68a7970bcc09bbbc891bdbc21047859a9e5"


@unittest.skipUnless(shutil.which("gfortran"), "GNU Fortran is required")
class ReductionLatchLifecycleTest(unittest.TestCase):
    def test_live_public_entry_clears_only_independent_call_failures(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                         FIXTURE_SHA256, "The sealed synthetic fixture changed")
        spec = importlib.util.spec_from_file_location("reduction_latch_fixture", FIXTURE)
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        source = LIVE.read_text()
        compiler = shutil.which("gfortran")
        started = time.monotonic()
        observations = {}
        counts = {"compiler": 0, "stub": 0}

        def run(argv, directory, kind, timeout):
            counts[kind] += 1
            self.assertLessEqual(counts[kind], 2 if kind == "compiler" else 16)
            remaining = 300 - (time.monotonic() - started)
            self.assertGreater(remaining, 0, "Inclusive synthetic-test deadline")
            result = subprocess.run(argv, cwd=directory,
                                    env=fixture.clean_environment(directory),
                                    capture_output=True, text=True,
                                    timeout=min(timeout, remaining), check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result.stdout

        with tempfile.TemporaryDirectory(prefix="madloop-latch-regression-",
                                         dir="/tmp") as temporary:
            for available in (False, True):
                directory = Path(temporary) / ("ninjars" if available else "plain")
                directory.mkdir()
                (directory / "stub.f").write_text(fixture.render(source, available))
                run([compiler, "-O0", "-fcheck=all", "-ffixed-line-length-none",
                     "-fno-fast-math", "stub.f", "-o", "check"],
                    directory, "compiler", 30)
                for case in range(1, 9):
                    observations[available, case] = fixture.parse(
                        run([str(directory / "check"), str(case)],
                            directory, "stub", 5))
        self.assertLessEqual(time.monotonic() - started, 300)
        self.assertEqual(counts, {"compiler": 2, "stub": 16})
        self.assertEqual(LIVE.read_text(), source, "Live source changed during test")

        for available in (False, True):
            for case in (1, 2, 3, 4):
                with self.subTest(ninjars=available, forced_mode=(1, 2, 4, 5)[case-1]):
                    first, second = observations[available, case]["rows"]
                    expected_failure = [1, 0] if case <= 2 else [0, 1]
                    expected_code = 110 if case <= 2 else 120
                    self.assertEqual(first["integers"][1], expected_code)
                    self.assertEqual(first["integers"][2:4], expected_failure)
                    self.assertEqual(first["integers"][4:6], [0, 0])
                    self.assertEqual(second["integers"][1], expected_code)
                    self.assertEqual(second["integers"][2:6], [0, 0, 0, 0],
                                     "A prior failed public call contaminated a healthy call")
                    self.assertEqual(first["values"], [7., 11., 13., 17., -1.])
                    self.assertEqual(second["values"], first["values"])

            within = observations[available, 5]
            self.assertEqual(within["rows"][0]["integers"][2:4], [1, 0])
            self.assertEqual(within["rows"][0]["integers"][6:], [3, 1, 1])
            self.assertEqual(len(within["groups"]), 6)
            self.assertTrue(all(group[4:] == [1, 0] for group in within["groups"]),
                            "A healthy group or internal 300/200 jump erased failure")

            automatic = observations[available, 6]
            self.assertEqual([row["integers"][1] for row in automatic["rows"]],
                             [323, 420, 219])
            self.assertEqual(automatic["rows"][0]["integers"][2:4], [0, 0])
            self.assertEqual(automatic["rows"][1]["integers"][3], 1)
            self.assertEqual(automatic["rows"][2]["integers"][2:6], [0, 0, 0, 0])
            self.assertEqual([group[3] for group in automatic["groups"]
                              if group[0] == 1 and group[2] == 1],
                             [1, 2, 1, 4, 5, 4])

            terminal_dp = observations[available, 7]["rows"]
            self.assertEqual([row["integers"][1] for row in terminal_dp], [410, 219])
            self.assertEqual(terminal_dp[0]["integers"][2:4], [1, 0])
            self.assertEqual(terminal_dp[1]["integers"][2:6], [0, 0, 0, 0])

            healthy = observations[available, 8]["rows"]
            self.assertEqual([row["integers"][1] for row in healthy],
                             [110, 110, 120, 120, 219])
            self.assertTrue(all(row["integers"][2:6] == [0, 0, 0, 0]
                                for row in healthy))
            self.assertTrue(all(row["values"][:4] == [7., 11., 13., 17.]
                                for row in healthy))
        for case in range(1, 9):
            self.assertEqual(observations[False, case], observations[True, case])
