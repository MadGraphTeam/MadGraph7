"""Ownership and startup-fallback checks of the generated provider routines."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TEMPLATES = ROOT/'madgraph/iolibs/template_files'

PREAMBLE = """
module mc_native_context
  implicit none
  integer, parameter :: provider_legs=2,provider_orders=2
  type BornModelState
    integer(8) :: configuration_epoch=1
  end type
  type BornResult
    integer :: id=0
  end type
  type BornLiveModel
    integer :: id=0
  end type
  integer :: active_context=0,active_sector=0,active_history=0,native_epoch=0
  integer :: shared_real_epoch=0,contexts(1),calls=0,native_calls=0,observed(8)=0
  integer,pointer :: native_metadata
  logical :: native_mapping=.false.,shared_real_active=.false.
  real(8) :: shared_real_point(0:3,2)
  type(BornResult) :: native_result
"""
HARNESS = """
  logical function born_model_state_equal(a,b)
    type(BornModelState) :: a,b
    born_model_state_equal=a%configuration_epoch==b%configuration_epoch
  end function
end module
subroutine mc_evaluate_provider_batch(p,g,e,row,flavour,n,total,orders,ok)
  use mc_native_context,only:calls,observed
  integer n,row,flavour,i
  real(8) p(0:3,2,n),g(n),e,total(n),orders(2,n)
  logical ok
  calls=calls+1
  observed(calls)=n
  ok=row==1
  if(.not.ok)return
  do i=1,n
    total(i)=p(0,1,i)*g(i)*e
    orders(:,i)=[total(i),0d0]
  enddo
end subroutine
subroutine mc_execute_native_fallback(q)
  use mc_native_context
  type(native_provider_request)::q
  native_calls=native_calls+1
  q%total=q%owner*100d0
  q%components=[q%total,0d0]
end subroutine
subroutine mc_native_oracle(q,total,orders)
  use mc_native_context
  type(native_provider_request)::q
  real(8)::total,orders(2)
  total=q%total
  orders=q%components
end subroutine
subroutine mc_commit_local_real(q)
  use mc_native_context
  type(native_provider_request)::q
  if(q%total/=q%components(1))error stop
end subroutine
program check
  use mc_native_context
  implicit none
  type(native_provider_request)::q(5)
  real(8)::total,orders(2)
  integer i
  character(12) mode
  call get_command_argument(1,mode)
  do i=1,5
    q(i)%owner=i
    q(i)%sequence=4
    q(i)%fold=2
    q(i)%row=1
    q(i)%flavour=1
    q(i)%g=i
    q(i)%e=2d0
    q(i)%momenta=i+10
  enddo
  ! First and last form a non-contiguous compatible group. The second
  ! request changes EW, the third has no provider, the fourth was a hit.
  q(2)%e=3d0
  q(3)%row=2
  q(4)%total=-1d0
  q(4)%components=[-1d0,0d0]
  q(4)%ready=.true.
  q(4)%immediate=.true.
  call mc_dispatch_provider_wave(q)
  if(any(.not.q%ready).or.calls/=3.or.any(observed(:3)/=[2,1,1]).or.native_calls/=0)error stop 1
  if(.not.q(3)%fallback)error stop 2
  if(mode=='swapped')call mc_consume_provider_result(q(1),total,orders,5_8,4,2)
  do i=1,5
    call mc_consume_provider_result(q(i),total,orders,int(i,8),4,2)
    if(i==3)then
      if(total/=300d0.or.native_calls/=1)error stop 3
    else if(i==4)then
      if(total/=-1d0)error stop 4
    else if(total/=(i+10)*i*q(i)%e)then
      error stop 5
    endif
  enddo
  if(mode=='duplicate')call mc_consume_provider_result(q(2),total,orders,2_8,4,2)
end program
"""


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class NativeProviderWave(unittest.TestCase):
    def test_compatible_compaction_owned_scatter_and_native_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            # Dead-code elimination removes the unrelated point-activation callbacks.
            declarations=PREAMBLE.replace('integer :: shared_real_epoch=0,contexts(1),','integer,target :: contexts(1)\n  integer :: shared_real_epoch=0,')
            code=(declarations+(TEMPLATES/'mc_native_provider_types.inc').read_text()+'\ncontains\n'+
                  (TEMPLATES/'mc_native_provider_methods.inc').read_text()+HARNESS)
            (work/'probe.f90').write_text(code)
            subprocess.run(['gfortran','-ffree-line-length-none','-ffunction-sections','-Wl,--gc-sections',
                            str(ROOT/'Template/NLO/SubProcesses/mc_batch_trace.f90'),'probe.f90','-o','probe'],
                           cwd=work,check=True,capture_output=True)
            subprocess.run([str(work/'probe')],check=True)
            for mode,reason in [('swapped','Wrong native real result owner'),('duplicate','repeated native real result')]:
                result=subprocess.run([str(work/'probe'),mode],capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn(reason,result.stderr)


    def test_opt_in_profile_counts_physical_inputs_and_native_routes(self):
        source = """
program check
  use iso_c_binding
  use mc_batch_trace
  implicit none
  real(c_double) p(8),g(2),result(2)
  integer(c_int32_t) flavour(2)
  character(16) mode
  call get_command_argument(1,mode)
  p=1d0
  g=1d0
  result=2d0
  flavour=0
  call mc_real_profile_open(2)
  call trace_umami_input(2_c_size_t,1,1,p,g,1d0,flavour)
  call trace_umami_output(2_c_size_t,1,result)
  call trace_fortran_result(2d0,result,2)
  if(mc_oracle_sample())call mc_oracle_report(1_8,1,2d0,result,2d0,result)
  if(mc_oracle_sample())error stop 'wrong sampling period'
  if(mode=='mixed')call mc_oracle_report(5_8,2,1d6,[1d6,1d-28],1d6,[1d6,0d0])
  call mc_real_profile_close()
end program
"""
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            (work/'probe.f90').write_text(source)
            subprocess.run(['gfortran',str(ROOT/'Template/NLO/SubProcesses/mc_batch_trace.f90'),
                            'probe.f90','-o','probe'],cwd=work,check=True)
            env={k:v for k,v in os.environ.items() if not k.startswith('MG7_MC_')}
            command=[str(work/'probe')]
            self.assertEqual(subprocess.check_output(command,env=env,text=True),'')
            env.update(MG7_MC_BATCH_PROFILE='1',MG7_MC_REAL_ORACLE_PERIOD='2')
            result=subprocess.check_output(command,env=env,text=True)
            self.assertIn('MG7_PROFILE_UMAMI mode/width/calls 2 2 1',result)
            self.assertIn('MG7_PROFILE_FORTRAN mode/configured/diagrams/unavailable 2 0 1 0',result)
            self.assertIn('MG7_REAL_ORACLE owner/row/status/provider/reference/maxratio 1 1 PASS',result)
            failure=subprocess.run(command+['mixed'],env=env,capture_output=True,text=True)
            self.assertNotEqual(failure.returncode,0)
            self.assertIn('5 2 FAIL',failure.stdout)
            env['MG7_MC_REAL_ORACLE_RECORD_ONLY']='1'
            result=subprocess.check_output(command+['mixed'],env=env,text=True)
            self.assertIn('5 2 FAIL',result)
            self.assertIn('MG7_REAL_ORACLE_COMPONENTS reference',result)
