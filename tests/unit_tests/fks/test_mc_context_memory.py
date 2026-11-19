"""Reservations cover the live MINT lane pool and deferred LHE text."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

TEMPLATE = Path(__file__).resolve().parents[3] / 'Template/NLO/SubProcesses'


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestNativeContextMemory(unittest.TestCase):
    def test_configured_ceiling_and_peak(self):
        program = """
program check
  use iso_fortran_env, only: int64
  use mc_context_memory
  use mc_event_records
  implicit none
  type(native_event_record), allocatable, target :: event
  character(20) :: mode
  call get_command_argument(1,mode)
  if(mode=='fixed')call configure_context_memory(2,600000_int64)
  call configure_context_memory(2,100000_int64)
  call enter_context(1)
  call reserve_context(weight_store,250000_int64)
  call enter_context(2)
  if(mode=='lane')call reserve_context(provider_store,700000_int64)
  if(mode=='grow')call reserve_context(weight_store,320000_int64,300000_int64)
  allocate(event)
  call begin_event_capture(event)
  call emit_event_line(event_capture_unit,'<event>')
  call emit_event_line(event_capture_unit,repeat('x',150000))
  if(mode=='event')call emit_event_line(event_capture_unit,repeat('y',700000))
  call emit_event_line(event_capture_unit,'  </event>')
  call end_event_capture(event)
  call release_contexts()
  ! The accepted record can outlive the entire cohort, then finalizes on discard.
  deallocate(event)
  call report_context_memory()
end program
"""
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / 'check.f90').write_text(program)
            subprocess.run(['gfortran','-O3','-fcheck=all', TEMPLATE/'mc_context_memory.f90',
                            TEMPLATE/'mc_event_records.f90', 'check.f90', '-o','check'], cwd=work, check=True)
            for mode, limit, ok in [('normal','1',True),('fixed','1',False),('lane','1',False),
                                    ('grow','1',False),('event','1',False),('normal','0',False),
                                    ('normal','10junk',False),('normal','999999999999999999',False)]:
                with self.subTest(mode=mode,limit=limit):
                    value = subprocess.run([work/'check',mode], cwd=work, capture_output=True,text=True,
                                           env={**os.environ,'MG7_NLO_BATCH_MEMORY_MIB':limit})
                    self.assertEqual(value.returncode == 0,ok,value.stdout+value.stderr)
                    if ok:
                        self.assertIn('peak reserved bytes=',value.stdout)
                        self.assertIn('largest live context reservation bytes=',value.stdout)
                    else:
                        self.assertIn('MG7_NLO_BATCH_MEMORY_MIB',value.stderr)
