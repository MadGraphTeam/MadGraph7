"""Native analytic limits and independent signed real insertions at the FKS boundary."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestNativeRealContinuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        inputs = {
            'nexternal.inc': ('integer nexternal,nincoming', 'parameter(nexternal=3,nincoming=2)'),
            'orders.inc': ('integer amp_split_size', 'parameter(amp_split_size=2)',
                           'double precision amp_split(2)', 'common/test_orders/amp_split'),
            'pmass.inc': ('pmass=0d0', 'pmass(3)=2d0'),
            'coupl.inc': (), 'fks_info.inc': (),
            'timing_variables.inc': ('logical timing_sampled', 'parameter(timing_sampled=.false.)',
                                     'integer timing_real','parameter(timing_real=1)'),
        }
        for name, lines in inputs.items():
            (cls.work/name).write_text(''.join('      ' + line + '\n' for line in lines))
        (cls.work/'phase.f90').write_text('module fks_phase_space_data\nreal(8) :: shat=400d0\nend module\n')
        singular = TEMPLATE/'fks_singular.f'
        routines = ('sreal', 'sreal_prepare', 'resolve_native_real_scalar')
        (cls.work/'boundary.f').write_text(''.join(fortran_routine(singular, name) for name in routines))
        (cls.work/'check.f90').write_text(PROGRAM)
        subprocess.run(['gfortran', '-O3', '-fno-fast-math', '-fcheck=all', '-ffixed-line-length-none',
                        '-I', str(cls.work), 'phase.f90', str(TEMPLATE/'mc_point_state.f'),
                        'boundary.f', 'check.f90', '-o', 'check'], cwd=cls.work, check=True,
                       capture_output=True, text=True)

    def test_independent_full_reals_and_native_limits(self):
        result = subprocess.run([str(self.work/'check')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('all native real boundary checks passed', result.stdout)

    def test_duplicate_and_missing_completion_fail(self):
        for mode in ('complete_twice', 'consume_twice', 'unready'):
            with self.subTest(mode=mode):
                result = subprocess.run([str(self.work/'check'), mode], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('native real', result.stderr.lower())


PROGRAM = """
program check
  use mc_point_state
  implicit none
  type(native_real_request) a,b
  real(8) p(0:3,3),q(0:3,3),out,w(2),a0(2),b0(2),split(2),original(2)
  real(8) xi_a,xi_b,y_a,y_b
  integer i_fks,j_fks,calls
  logical softtest,colltest,colour,charge
  character(20) mode
  common/test_orders/split
  common/fks_indices/i_fks,j_fks
  common/sctests/softtest,colltest
  common/c_need_links/colour,charge
  common/test_calls/calls
  calls=0
  softtest=.false.; colltest=.false.; colour=.true.;charge=.false.
  i_fks=3; j_fks=1
  p=0d0; p(0,1:2)=10d0; p(3,1)=10d0; p(3,2)=-10d0; p(0,3)=9d0
  q=p; q(0,3)=13d0
  xi_a=0.39d0; xi_b=0.9d0; y_a=0.761d0; y_b=-0.4d0
  a0=[-14.61d0,107.2d0]; b0=[2.30d0,-1.1d0]
  call sreal_prepare(p,xi_a,y_a,out,a)
  call sreal_prepare(q,xi_b,y_b,out,b)
  if(a%status/=real_waiting.or.b%status/=real_waiting.or.calls/=0)error stop 1
  if(any(a%p/=p).or.any(b%p/=q))error stop 2
  call get_command_argument(1,mode)
  if(mode=='unready')call consume_native_real(a,out,w)
  call complete_native_real(b,sum(b0),b0)
  call complete_native_real(a,sum(a0),a0)
  if(mode=='complete_twice')call complete_native_real(a,sum(b0),b0)
  call consume_native_real(a,out,w)
  if(mode=='consume_twice')call consume_native_real(a,out,w)
  if(out/=(sum(a0)*xi_a**2)*(1d0-y_a).or.any(w/=(a0*xi_a**2)*(1d0-y_a)))error stop 3
  call consume_native_real(b,out,w)
  if(out/=(sum(b0)*xi_b**2)*(1d0-y_b).or.any(w/=(b0*xi_b**2)*(1d0-y_b)))error stop 4
  call sreal(q,xi_b,y_b,out)
  if(calls/=1.or.any(split/=([26d0,-13d0]*xi_b**2)*(1d0-y_b)))error stop 5
  call sreal_prepare(p,0d0,0.2d0,out,a)
  if(a%status/=real_idle.or.out/=7d0.or.any(split/=[3d0,4d0]))error stop 6
  call sreal_prepare(p,0.4d0,1d0,out,a)
  if(a%status/=real_idle.or.out/=11d0)error stop 7
  j_fks=3
  call sreal_prepare(p,0.4d0,1d0,out,a)
  if(a%status/=real_idle.or.out/=0d0.or.any(split/=0d0))error stop 8
  softtest=.true.
  call sreal_prepare(p,1d-9,0.4d0,out,a)
  if(a%status/=real_waiting)error stop 9
  call resolve_native_real_scalar(a)
  call consume_native_real(a,out,w)
  if(calls/=2.or.any(w/=([18d0,-9d0]*1d-18)*(1d0-0.4d0)))error stop 10
  p(0,1)=-1d0; original=split
  call sreal_prepare(p,0.4d0,0.5d0,out,a)
  if(a%status/=real_idle.or.out/=0d0.or.any(original/=split))error stop 11
  print *,'all native real boundary checks passed'
end
real(8) function dot(p,q)
  real(8) p(0:3),q(0:3)
  dot=p(0)*q(0)-dot_product(p(1:3),q(1:3))
end
subroutine smatrix_real_frame(p,w)
  integer calls
  real(8) p(0:3,3),w,split(2)
  common/test_orders/split
  common/test_calls/calls
  calls=calls+1; split=[2*p(0,3),-p(0,3)]; w=sum(split)
end
subroutine sbornsoft(p,xi,y,w)
  real(8) p(0:3,3),xi,y,w,split(2)
  common/test_orders/split
  split=[3d0,4d0];w=sum(split)
end
subroutine sborncol_isr(p,xi,y,w)
  real(8) p(0:3,3),xi,y,w,split(2)
  common/test_orders/split
  split=[5d0,6d0];w=sum(split)
end
subroutine sborncol_fsr(p,xi,y,w)
  real(8) p(0:3,3),xi,y,w
  error stop 'unexpected final-state collinear call'
end
"""
