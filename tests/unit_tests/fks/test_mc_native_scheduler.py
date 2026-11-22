"""Unequal request waves, accepted-event ownership, and terminal surplus."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

TEMPLATE = Path(__file__).resolve().parents[3] / 'Template/NLO/SubProcesses'


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestNativeScheduler(unittest.TestCase):
    def test_waiting_lanes_close_in_order_and_queues_keep_their_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, contents in [('support.f90', SUPPORT), ('check.f90', CHECK)]:
                (root / name).write_text(contents)
            sources = [TEMPLATE / 'mc_context_memory.f90', TEMPLATE / 'mc_decision_ledger.f90', root / 'support.f90',
                       TEMPLATE / 'mc_native_cohort.f90', TEMPLATE / 'mc_event_queues.f90',
                       TEMPLATE.parent / 'Source/timing.f', root / 'check.f90']
            subprocess.run(['gfortran', '-O3', '-fno-fast-math', '-fcheck=all', '-I', str(TEMPLATE),
                            *sources, '-o', 'check'], cwd=root, check=True, capture_output=True)
            subprocess.run([root / 'check'], cwd=root, check=True, capture_output=True)


SUPPORT = """
module mc_native_context
  implicit none
  type native_provider_request
     integer :: key=0,result=0
  end type
  integer :: calls(3)=0
contains
  subroutine mc_dispatch_provider_wave(requests)
    type(native_provider_request) :: requests(:)
    integer i
    calls(size(requests))=calls(size(requests))+1
    do i=1,size(requests)
       requests(i)%result=3*requests(i)%key
    enddo
  end subroutine
end module

module mc_native_points
  use iso_fortran_env, only: int64
  use mc_native_context
  implicit none
  integer,parameter :: mc_wait_real=1,mc_fold_complete=0
  type native_mc_point
     integer(int64) :: draw_id=0
     integer :: received=0,status=mc_fold_complete
     logical :: has_chart=.true.
     type(native_provider_request) :: provider
  end type
  integer :: active=0,closed=0,recycled=0,settled=0,released=0
contains
  subroutine settle_native_points()
    if (active/=0) error stop 'Settled while a point is active'
    settled=settled+1
  end subroutine
  subroutine release_native_points()
    if (active/=0) error stop 'Released while a point is active'
    released=released+1
  end subroutine
  subroutine recycle_native_point(point)
    type(native_mc_point) :: point
    ! Weight-line stores move through the active arrays: no point may be live.
    if (active/=0) error stop 'Recycled a point while another is active'
    recycled=recycled+1
  end subroutine
  subroutine activate_native_point(point)
    type(native_mc_point) :: point
    if (active/=0) error stop 'Interfering active point'
    active=point%draw_id
  end subroutine
  subroutine park_native_point(point)
    type(native_mc_point) :: point
    if (active/=point%draw_id) error stop 'Parking wrong point'
    if (closed>=active.and.point%has_chart) error stop 'Parked a closed subtraction chart'
    active=0
  end subroutine
end module

module mint_module
  use iso_fortran_env, only: int64
  implicit none
  integer :: imode=2,surplus=0
  type mint_draw
     integer(int64) :: id
     integer :: ifl=0
     logical :: complete=.false.
     real(8) :: coordinates(3),volume=1d0,integrals(2)=0d0
  end type
contains
  subroutine activate_mint_draw(draw)
    type(mint_draw) :: draw
  end subroutine
  subroutine park_mint_draw(draw)
    type(mint_draw) :: draw
  end subroutine
  subroutine finish_mint_fold(draw)
    type(mint_draw) :: draw
    draw%complete=draw%ifl==2
    draw%ifl=2
  end subroutine
  subroutine trace_mc_draw(action)
    character(*) action
    if (action/='SURPLUS') error stop
    surplus=surplus+1
  end subroutine
end module

module mc_event_records
  use iso_fortran_env, only: int64
  type native_event_record
     integer(int64) :: draw_id
     integer,allocatable :: colours(:)
  end type
end module

subroutine sigintF_advance(point,x,vol,ifl,f)
  use mc_native_points
  type(native_mc_point) :: point
  real(8) :: x(3),vol,f(2)
  integer :: ifl
  if (active/=point%draw_id) error stop 'Callback owner'
  if (ifl==2) then
     closed=closed+1
     if (closed/=active) error stop 'Out-of-order fit'
     f(2)=point%received
     return
  endif
  if (point%status==mc_wait_real) then
     if (point%provider%result/=3*(100*active+point%received)) error stop 'Wrong returned result'
     point%received=point%received+1
  endif
  if (point%received<active-1) then
     point%provider%key=100*active+point%received
     point%status=mc_wait_real
  else
     point%status=mc_fold_complete
  endif
end subroutine
"""

CHECK = """
program check
  use mc_native_points
  use mc_native_cohort
  use mc_event_queues
  use mc_event_records
  use mint_module
  use mc_decision_ledger, only: current_draw
  implicit none
  type(mint_draw) :: draws(3)
  type(event_queue) :: queues(4)
  type(native_event_record),allocatable :: event
  integer :: i,stream
  call timing_reset()
  do i=1,3
     draws(i)%id=i
  enddo
  call evaluate_native_cohort(draws)
  if (any(calls/=[1,1,0]).or..not.all(draws%complete)) error stop 'Wave widths'
  do i=1,3
     if (draws(i)%integrals(2)/=i-1) error stop 'Wrong fold result'
     call restore_generation_draw(draws(i),i)
     allocate(event)
     event%draw_id=draws(i)%id
     event%colours=[701+i,711+i]
     stream=generation_stream('virt',merge(2,1,i==2))
     call append_event(queues(stream),event,3)
     if (allocated(event)) error stop 'Duplicate ownership'
     call park_generation_draw(i)
  enddo
  call release_native_cohort()
  if (recycled/=3) error stop 'Released lanes kept their stores'
  if (settled/=1.or.released/=1) error stop 'Parked state not settled before release'
  if (generation_stream('novi',2)==generation_stream('virt',2)) error stop 'Envelope conflated with stream'
  if (generation_stream('born',3)/=4) error stop 'Born stream'
  call take_event(queues(2),event)
  if (current_draw/=2.or.any(event%colours/=[703,713])) error stop 'Wrong deferred event'
  call take_event(queues(1),event)
  if (current_draw/=1.or.any(event%colours/=[702,712])) error stop 'Wrong FIFO order'
  call discard_surplus(queues)
  if (current_draw/=3.or.surplus/=1.or.queues(1)%surplus/=1) error stop 'Unaccounted terminal event'
end program
"""
