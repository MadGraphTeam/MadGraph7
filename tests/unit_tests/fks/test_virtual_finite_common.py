"""Keep the per-order virtual return visible to the native FKS consumer.

The frame wrapper is production code; only the expensive loop/Born evaluations
are deterministic stubs. The consumer's COMMON declaration and assignment are
taken from bornsoftvirtual, so a correct total virtual alone cannot pass.
"""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from madgraph.iolibs import born_support
from tests.unit_tests.fks.test_momentum_maps import TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestVirtualFiniteCommon(unittest.TestCase):
    def check_frame_return(self, native):
        with tempfile.TemporaryDirectory(prefix='mg7_virtual_finite_') as tmp:
            work = Path(tmp)
            bridge = work / 'boost_to_frame.f'
            shutil.copyfile(TEMPLATE / bridge.name, bridge)
            if native:
                born_support.write_native_frame_bridge(bridge)
            (work / 'wrapper.f').write_text(
                fortran_routine(bridge, 'binothlha_frame'))

            # Use the actual caller's storage and read, independently of the
            # declaration in the wrapper under test.
            consumer = born_support.routine(
                (TEMPLATE / 'fks_singular.f').read_text(), 'bornsoftvirtual')
            declaration = next(line for line in consumer if re.match(
                r'\s*double precision amp_split_finite_ML\(', line, re.I))
            common = next(line for line in consumer if re.match(
                r'\s*common\s*/to_amp_split_finite/', line, re.I))
            read = next(line for line in consumer if re.match(
                r'\s*amp_split_virt\(iamp\)\s*=\s*amp_split_finite_ML\(iamp\)',
                line, re.I))
            born_support.write_fortran(work / 'consumer.f', [
                'subroutine read_finite(amp_split_virt,reset)',
                'implicit none', "include 'orders.inc'", declaration, common,
                'double precision amp_split_virt(amp_split_size)',
                'logical reset', 'integer iamp',
                'if(reset)amp_split_finite_ML=-999d0',
                'do iamp=1,amp_split_size', read, 'enddo', 'end'])
            for name, lines in {
                'nexternal.inc': ['integer,parameter::nexternal=5,nincoming=2'],
                'genps.inc': ['integer,parameter::ngraphs=1,ncolor=1'],
                'orders.inc': ['integer,parameter::amp_split_size=2,nsplitorders=2'],
                'born_nhel.inc': ['integer,parameter::max_bhel=1'],
            }.items():
                born_support.write_fortran(work / name, lines)
            (work / 'check.f90').write_text(
                DRIVER.replace('@NATIVE@', '.true.' if native else '.false.'))
            build = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all',
                '-ffixed-line-length-none', '-ffree-line-length-none',
                'check.f90', 'wrapper.f', 'consumer.f', '-o', 'check'],
                cwd=work, capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(work / 'check')], cwd=work,
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('PASS per-order finite virtual', run.stdout)

    def test_frame_wrapper_returns_the_finite_orders(self):
        self.check_frame_return(native=False)

    def test_native_rewrite_preserves_the_finite_orders(self):
        self.check_frame_return(native=True)


DRIVER = '''module mc_born_types
implicit none
integer :: born_active_helicity_frame(2)=0
end module

module mc_native_context
implicit none
integer :: active_context=23,ensure_count=0
contains
subroutine ensure_native_context()
ensure_count=ensure_count+1
end subroutine
end module

module virtual_inputs
implicit none
integer :: selected_ids(4)=0,expected_key(2)=0,calls=0
real(8) :: expected_orders(2),expected_energy
end module

program check_finite
use mc_born_types
use mc_native_context
use virtual_inputs
implicit none
logical,parameter :: native=@NATIVE@
real(8) :: p(0:3,4),virtual,finite(2)
integer phase
do phase=1,2
  p=real(phase,8)
  expected_orders=[3.5d0*phase,-0.75d0*phase**2]
  expected_energy=p(0,1)
  selected_ids=0
  born_active_helicity_frame=[77,88]
  expected_key=born_active_helicity_frame
  if(phase.eq.2)then
    selected_ids=[0,0,1,0]
    expected_energy=expected_energy+5d0
    if(native)expected_key=[23,8]
  endif
  call read_finite(finite,.true.)
  call binothlha_frame(p,2d0,virtual)
  if(virtual.ne.sum(expected_orders))error stop 'incorrect total virtual'
  call read_finite(finite,.false.)
  if(any(finite.ne.expected_orders))error stop 'lost finite virtual orders'
  if(any(born_active_helicity_frame.ne.[77,88]))error stop 'frame scope leaked'
  if(any(p.ne.real(phase,8)))error stop 'input momenta changed'
enddo
if(calls.ne.2)error stop 'missing virtual call'
if(native.and.ensure_count.ne.2)error stop 'native context not activated'
print *, 'PASS per-order finite virtual'
end program

subroutine get_frame_mask_born(ids)
use virtual_inputs
implicit none
integer ids(4)
ids=selected_ids
end subroutine

subroutine boost_to_me_frame(p,n,ids,pf)
implicit none
integer n,ids(n)
real(8) p(0:3,n),pf(0:3,n)
pf=p
if(any(ids.ne.0))pf(0,1)=pf(0,1)+5d0
end subroutine

subroutine sborn_amp(p,amp2,jamp2,split,cnts,born,ans,saved)
implicit none
real(8) p(0:3,4),amp2(1),jamp2(0:1),split(2),born
complex(8) cnts(2,2,2),ans(2,2),saved(1,1)
amp2=0d0
jamp2=0d0
split=[0.75d0,-0.25d0]
cnts=0d0
born=2d0
ans=0d0
saved=cmplx(1d0,2d0,8)
end subroutine

subroutine BinothLHA(p,born,virtual,split,saved,finite,cnts)
use mc_born_types
use virtual_inputs
implicit none
real(8) p(0:3,4),born,virtual,split(2),finite(2)
complex(8) saved(1,1),cnts(2,2,2)
if(p(0,1).ne.expected_energy)error stop 'wrong loop evaluation frame'
if(any(born_active_helicity_frame.ne.expected_key))error stop 'wrong helicity frame'
if(any(split.ne.[0.75d0,-0.25d0]))error stop 'wrong Born split return'
if(saved(1,1).ne.cmplx(1d0,2d0,8))error stop 'wrong Born amplitude return'
finite=expected_orders
virtual=sum(finite)
calls=calls+1
end subroutine
'''
