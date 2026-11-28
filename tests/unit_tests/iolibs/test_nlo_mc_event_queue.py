"""Compile the production accepted-event service with controlled hit/miss decisions."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from madgraph import MG5DIR

TEMPLATE = Path(MG5DIR) / 'Template/NLO/SubProcesses'


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestMCEventQueue(unittest.TestCase):
    def test_empty_waves_stream_switches_owned_metadata_and_terminal_surplus(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work/'trials.f90').write_text(TRIALS)
            (work/'check.f90').write_text(CHECK)
            subprocess.run(['gfortran', '-O3', '-fno-fast-math', '-fcheck=all',
                            TEMPLATE/'mc_decision_ledger.f90', 'trials.f90',
                            TEMPLATE/'mc_context_memory.f90', TEMPLATE/'mc_event_records.f90', TEMPLATE/'mc_event_queues.f90',
                            TEMPLATE/'mc_native_generation.f90', 'check.f90', '-o', 'check'],
                           cwd=work, capture_output=True, check=True)
            result = subprocess.run([work/'check'], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            expected = []
            for draw in (4, 7, 11, 6, 8, 13):
                expected.extend(['  <event>', f'{draw} {100*draw} {700+draw}   ',
                                 'x' * 1024, str(draw % 10) * 3000, '  </event>'])
            self.assertEqual((work/'served.lhe').read_text(), '\n'.join(expected)+'\n')
            self.assertIn('SURPLUS 9', result.stdout)
            self.assertIn('SURPLUS 15', result.stdout)


TRIALS = """
module mint_module
  use iso_fortran_env, only: int64
  use mc_decision_ledger, only: current_draw
  implicit none
  integer,parameter :: ndimmax=3
  type mint_draw
     integer(int64) :: id
  end type
  integer :: wave=0,requests(3)=0,outputs(3)=0
contains
  subroutine increase_gen_counters_before(vn)
    integer vn
    requests(vn)=requests(vn)+1
  end subroutine
  subroutine increase_gen_counters_end(vn)
    integer vn
    outputs(vn)=outputs(vn)+1
  end subroutine
  subroutine trace_mc_draw(action)
    character(*) action
    write(*,'(a,1x,i0)') action,current_draw
  end subroutine
  subroutine generate_mc_trials(draws,evaluate,vn,accepted)
    type(mint_draw),intent(inout) :: draws(:)
    interface
       subroutine evaluate(draws)
         import mint_draw
         type(mint_draw),intent(inout) :: draws(:)
       end subroutine
    end interface
    integer vn,lane
    logical,intent(out) :: accepted(:)
    if (size(draws)/=3) error stop 'unexpected capacity'
    wave=wave+1
    do lane=1,3
       draws(lane)%id=3*(wave-1)+lane
    enddo
    call evaluate(draws)
    select case(wave)
    case(1)
       accepted=.false.
    case(2,5)
       accepted=[.true.,.false.,.true.]
    case(3)
       accepted=.true.
    case(4)
       accepted=[.false.,.true.,.false.]
    case default
       error stop 'drew new candidates while a matching event was queued'
    end select
  end subroutine
end module

module mc_native_cohort
  use mint_module, only: mint_draw
  implicit none
  integer :: active=0,scale=0,partner=0
contains
  subroutine evaluate_native_cohort(draws)
    type(mint_draw),intent(inout) :: draws(:)
  end subroutine
  subroutine restore_generation_draw(draw,slot)
    type(mint_draw) :: draw
    integer slot
    if (active/=0) error stop 'previous owner still active'
    active=draw%id
    scale=100*active
    partner=700+active
  end subroutine
  subroutine park_generation_draw(slot)
    integer slot
    active=0
    scale=-1
    partner=-2
  end subroutine
  subroutine release_native_cohort()
    if (active/=0) error stop 'live weight lines on release'
  end subroutine
end module
"""

CHECK = """
subroutine select_native_event(requested,x,w)
  use mc_native_cohort
  use mint_module, only: ndimmax
  real(8) x(ndimmax),requested,w
  x=[dble(active),dble(scale),dble(partner)]
  w=requested
end subroutine
subroutine capture_event_record(record,x,w,putonshell,p_label)
  use mc_event_records, only: native_event_record,begin_event_capture,end_event_capture, &
       emit_event_line,event_capture_unit
  use mc_decision_ledger, only: current_draw
  implicit none
  type(native_event_record),target,intent(inout) :: record
  real(8),intent(in) :: x(*),w
  logical,intent(in) :: putonshell
  integer,intent(in) :: p_label
  character(80) line
  if (int(x(1))/=current_draw.or.w/=1d0.or..not.putonshell.or.p_label/=4) error stop
  call begin_event_capture(record)
  call emit_event_line(event_capture_unit,'  <event>')
  write(line,'(i0,2(1x,i0))') int(x(1)),int(x(2)),int(x(3))
  call emit_event_line(event_capture_unit,trim(line)//'   ')
  call emit_event_line(event_capture_unit,repeat('x',1024))
  call emit_event_line(event_capture_unit,repeat(achar(48+mod(int(x(1)),10)),3000))
  call emit_event_line(event_capture_unit,'  </event>')
  call end_event_capture(record)
end subroutine

program check
  use mint_module, only: wave,requests,outputs
  use mc_native_generation
  implicit none
  integer unit,i
  character(4),parameter :: kind(6)=['novi','virt','virt','novi','virt','born']
  integer,parameter :: vn(6)=[2,1,2,2,1,3]
  open(newunit=unit,file='served.lhe',status='new')
  do i=1,6
     call write_requested_event(kind(i),vn(i),3,1d0,.true.,4,unit)
  enddo
  call finish_native_generation()
  close(unit)
  if (wave/=5.or.any(outputs/=[2,3,1]).or.any(outputs/=requests)) error stop 'wrong number of events'
end program
"""
