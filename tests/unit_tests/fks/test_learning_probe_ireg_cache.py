"""Memoized integrated soft scalars must retain every physical kernel input."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import TEMPLATE, fortran_routine


PROGRAM = '''program check
use FKSParams,only: FKSSectorLearning
use fks_phase_space_data,only: shat
use kernel_counter
implicit none
real(8) p(0:3,7),pmass(7),QES2,xicut,expected,actual
common/to_mass/pmass
common/qes/QES2
character(4) abrv
common/to_abrv/abrv
integer i,j,before
p=0d0
p(:,1)=[1000d0,0d0,0d0,1000d0]
p(:,2)=[1000d0,0d0,0d0,-1000d0]
p(:,3)=[500d0,sqrt(500d0**2-173d0**2),0d0,0d0]
p(:,4)=[500d0,-p(1,3),0d0,0d0]
p(:,5)=[500d0,0d0,500d0,0d0]
p(:,6)=[500d0,0d0,-500d0,0d0]
pmass=[0d0,0d0,173d0,173d0,0d0,0d0,0d0]
shat=4d6
QES2=10000d0
xicut=0.1d0
abrv='all'
FKSSectorLearning=.true.
call reset_folded_virtual
! Cover all massless, mixed-mass, massive-pair and massive-self links.
do i=1,6
 do j=1,i
  if(i.eq.j.and.pmass(i).eq.0d0)cycle
  call raw_eikonal_Ireg(p,i,j,xicut,expected)
  before=kernel_calls
  call cached_eikonal_Ireg(p,i,j,xicut,actual)
  if(actual.ne.expected.or.kernel_calls.ne.before+1)error stop 'uncached link changed'
  call cached_eikonal_Ireg(p,i,j,xicut,actual)
  if(actual.ne.expected.or.kernel_calls.ne.before+1)error stop 'link not reused'
 enddo
enddo
if(kernel_calls.ne.17)error stop 'wrong link count'
! Every key field is changed independently, with a miss then a hit.
call miss_and_hit(.false.)
p(0,1)=p(0,1)+1d0
call miss_and_hit(.true.)
pmass(3)=174d0
call miss_and_hit(.true.)
shat=4.1d6
call miss_and_hit(.true.)
QES2=11000d0
call miss_and_hit(.true.)
xicut=0.15d0
call miss_and_hit(.true.)
abrv='grid'
call miss_and_hit(.true.)
! A new full point must drop even a matching physical-state table.
call reset_folded_virtual
call miss_and_hit(.true.)
FKSSectorLearning=.false.
before=kernel_calls
call cached_eikonal_Ireg(p,3,1,xicut,actual)
call cached_eikonal_Ireg(p,3,1,xicut,actual)
if(kernel_calls.ne.before+2)error stop 'reuse switch ignored'
call raw_eikonal_Ireg(p,3,1,xicut,expected)
if(actual.ne.expected)error stop 'disabled cache changed result'
write(*,*)'PASS integrated soft cache'
contains
subroutine miss_and_hit(miss)
logical miss
integer expected_calls
before=kernel_calls
expected_calls=before
if(miss)expected_calls=expected_calls+1
call raw_eikonal_Ireg(p,3,1,xicut,expected)
call cached_eikonal_Ireg(p,3,1,xicut,actual)
if(actual.ne.expected.or.kernel_calls.ne.expected_calls)error stop 'cache guard missed changed input'
call cached_eikonal_Ireg(p,3,1,xicut,actual)
if(actual.ne.expected.or.kernel_calls.ne.expected_calls)error stop 'unchanged scalar not cached'
end subroutine
end program
'''


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestLearningProbeIregCache(unittest.TestCase):
    def test_physical_link_table_and_all_scalar_input_guards(self):
        with tempfile.TemporaryDirectory(prefix='mg5_fold_ireg_') as tmp:
            work = Path(tmp)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\nparameter(nexternal=7,nincoming=2)',
                'coupl.inc': '',
                'orders.inc': 'integer nsplitorders\nparameter(nsplitorders=1)',
                'q_es.inc': 'real(8) QES2\ncommon/qes/QES2',
                'pmass.inc': 'call fixture_external_masses(pmass)',
            }
            for name, body in includes.items():
                (work / name).write_text(''.join('      '+line+'\n' for line in body.splitlines()))
            source = ''.join(fortran_routine(TEMPLATE / 'fks_singular.f', name)
                             for name in ('cached_eikonal_Ireg', 'reset_folded_virtual', 'xj1a', 'ddilog'))
            source += fortran_routine(TEMPLATE / 'fks_singular.f', 'eikonal_Ireg').replace(
                'subroutine eikonal_Ireg(', 'subroutine raw_eikonal_Ireg(', 1)
            source += fortran_routine(TEMPLATE.parent / 'Source/kin_functions.f', 'dot')
            (work / 'kernels.f').write_text(source)
            (work / 'counter.f90').write_text('''module kernel_counter
integer :: kernel_calls=0
end module
subroutine fixture_external_masses(pmass)
implicit none
real(8) pmass(7),active_mass(7)
common/to_mass/active_mass
pmass=active_mass
end subroutine
subroutine eikonal_Ireg(p,m,n,xicut,result)
use kernel_counter
implicit none
real(8) p(0:3,7),xicut,result
integer m,n
kernel_calls=kernel_calls+1
call raw_eikonal_Ireg(p,m,n,xicut,result)
end subroutine
''')
            (work / 'check.f90').write_text(PROGRAM)
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-ffixed-line-length-none', '-fcheck=all',
                '-ffpe-trap=invalid,zero,overflow', '-I', str(work),
                str(TEMPLATE / 'FKSParams.f90'), str(TEMPLATE / 'fks_phase_space_data.f'),
                str(work / 'counter.f90'), str(work / 'kernels.f'), str(work / 'check.f90'),
                '-o', str(work / 'check')], cwd=work, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(work / 'check')], cwd=work,
                                    text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS integrated soft cache', result.stdout)
