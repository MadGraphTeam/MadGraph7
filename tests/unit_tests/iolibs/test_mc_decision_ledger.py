"""Verify that interleaving two native draws preserves each recorded decision."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which("gfortran"), "requires gfortran")
class TestMCDecisionLedger(unittest.TestCase):
    def test_interleaved_replay_and_mismatched_ownership(self):
        program = """
program exercise
  use mc_decision_ledger
  use iso_fortran_env, only: int64
  implicit none
  double precision :: r
  character(16) :: action
  call get_command_argument(1,action)
  call decision_open()
  call decision_begin_draw()
  if (action=='record') then
    r=0.25d0
    call decision_uniform(3,r)
    call decision_context(1_int64,1,stage_physics)
    r=0.125d0
    call decision_uniform(3,r)
    call decision_begin_draw()
    r=0.75d0
    call decision_uniform(8,r)
  else
    r=-1d0
    call decision_begin_draw()
    call decision_uniform(8,r)
    if (r/=0.75d0) error stop 'wrong B'
    call decision_context(1_int64,0,stage_coordinates)
    call decision_uniform(3,r)
    if (r/=0.25d0) error stop 'wrong A'
    call decision_context(1_int64,1,stage_physics)
    if (action=='fold') call decision_context(1_int64,2,stage_physics)
    if (action/='omit') call decision_uniform(3,r)
    if (action=='extra') call decision_uniform(3,r)
    if (action=='replay'.and.r/=0.125d0) error stop 'wrong second A'
  endif
  call decision_close()
end program
"""
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            driver = work / "exercise.f90"
            driver.write_text(program)
            exe = work / "exercise"
            ledger = work / "rng.txt"
            subprocess.run(["gfortran", "-O3", "-fno-fast-math", "-fcheck=all",
                            ROOT / "Template/NLO/SubProcesses/mc_decision_ledger.f90",
                            driver, "-o", exe], cwd=work, check=True)
            env = dict(os.environ, MG7_MC_DECISION_FILE=str(ledger))
            subprocess.run([exe, "record"], env=dict(env, MG7_MC_DECISIONS="record"), check=True)
            env["MG7_MC_DECISIONS"] = "replay"
            subprocess.run([exe, "replay"], env=env, check=True)
            for action, message in [("fold", "another fold"), ("extra", "Extra stochastic"),
                                    ("omit", "Unconsumed native")]:
                with self.subTest(action=action):
                    failed = subprocess.run([exe, action], env=env, capture_output=True, text=True)
                    self.assertNotEqual(failed.returncode, 0)
                    self.assertIn(message, failed.stderr)
            ledger.write_text(ledger.read_text().split("END ")[0])
            failed = subprocess.run([exe, "replay"], env=env, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Missing end of native decision ledger", failed.stderr)
