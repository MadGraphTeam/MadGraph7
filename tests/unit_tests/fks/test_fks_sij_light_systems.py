"""FKS S functions of resolved soft partons in light systems.

The legacy external dot product rounds |p.q| < 1d-6 GeV**2 to zero. For
g g > h g at sqrt(shat)=3.1 GeV a gluon with xi ~ 1d-6, just above the
fks_Sij threshold where the real momenta replace the soft direction, has
products of that size with the beams. With that dot the real event put the
gluon on a beam axis (S=1 or S=0) while its soft counterterm kept the true
partition, so the real and soft weights no longer cancelled.
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.unit_tests.fks.test_momentum_maps import fortran_routine

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / 'Template/NLO/SubProcesses'

DRIVER = '''program check_light_soft_sij
use fks_phase_space_data, only: shat,sqrtshat,ybst_til_tocm,p_i_fks_cnt
implicit none
include 'nexternal.inc'
integer :: partners(nexternal,0:nexternal),types(nexternal),pdgs(nexternal),ifks,jfks
common/c_fks_inc/partners,types,pdgs
common/fks_indices/ifks,jfks
logical :: radiation(nexternal),charged(nexternal)
common/c_is_aorg/radiation
common/c_is_charged/charged
real(8) :: pmass(nexternal)
common/to_mass/pmass
real(8),parameter :: costh(6)=[0.999d0,0.97d0,0.88d0,0.3d0,-0.5d0,-0.95d0]
real(8),parameter :: xis(4)=[1.5d-6,3d-6,1d-5,1d-4],b=1.5d0
real(8) :: p(0:3,nexternal),q(0:3,nexternal),n(0:3),c,s,xi
real(8) :: real1,real2,soft1,expected,fks_Sij
integer :: ia,ix
pdgs=[21,21,25,21]
types=[8,8,1,8]
radiation=[.true.,.true.,.false.,.true.]
charged=.false.
pmass=[0d0,0d0,3.1d0,0d0]
partners=0
partners(4,0)=2
partners(4,1:2)=[1,2]
ifks=4
sqrtshat=3.1d0
shat=sqrtshat**2
! Real and Born frames differ by a tiny boost, as for the ISR maps: this
! selects the covariant energies and angles of get_cms_costh_fks.
ybst_til_tocm=-1d-6
do ia=1,size(costh)
  c=costh(ia)
  s=sqrt((1d0-c)*(1d0+c))
  n=[1d0,s*cos(.7d0),s*sin(.7d0),c]
  p_i_fks_cnt(:,0)=sqrtshat/2d0*n
  p_i_fks_cnt(:,2)=p_i_fks_cnt(:,0)
  expected=1d0/(1d0+((1d0-c)/(1d0+c))**b)
  do ix=1,size(xis)
    xi=xis(ix)
    p=0d0
    p(:,1)=sqrtshat/2d0*[1d0,0d0,0d0,1d0]
    p(:,2)=sqrtshat/2d0*[1d0,0d0,0d0,-1d0]
    p(:,4)=xi*sqrtshat/2d0*n
    p(:,3)=p(:,1)+p(:,2)-p(:,4)
    q=p
    q(:,4)=0d0
    jfks=1
    real1=fks_Sij(p,4,1,xi,c)
    soft1=fks_Sij(q,4,1,0d0,c)
    jfks=2
    real2=fks_Sij(p,4,2,xi,-c)
    if(abs(real1-expected).gt.1d-9.or.abs(soft1-expected).gt.1d-9.or. &
         abs(real1+real2-1d0).gt.1d-9)then
      write(*,*) 'cos, xi, S_real, S_soft, expected, S_41+S_42',c,xi,real1,soft1,expected,real1+real2
      error stop 'soft S partition of a light system'
    endif
  enddo
enddo
print *, 'PASS light-system soft S partition'
end program
'''


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestFKSSijLightSystems(unittest.TestCase):
    """The S partition of a resolved soft parton matches its soft limit."""

    def test_resolved_soft_gluon_keeps_its_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for name, source in {
                    'nexternal.inc': 'integer nexternal,nincoming\n'
                                     'parameter(nexternal=4,nincoming=2)',
                    'coupl.inc': '',
                    'pmass.inc': 'common/to_mass/pmass'}.items():
                (work / name).write_text(''.join('      ' + line + '\n'
                                                 for line in source.splitlines()))
            shutil.copyfile(TEMPLATE / 'fks_powers.inc', work / 'fks_powers.inc')
            routines = [fortran_routine(TEMPLATE / 'fks_Sij.f', name) for name in
                        ('fks_Sij', 'get_dkl_Sij', 'get_cms_energy',
                         'get_cms_costh_fks', 'costh_fks', 'h_damp')]
            # Only other helpers of fks_phase_space_helpers need the legacy dot.
            routines.append(fortran_routine(ROOT / 'Template/NLO/Source/kin_functions.f', 'dot'))
            # fks_Sij leaves bound states out of the partition (onia_legs.f); no onia here
            routines.append(fortran_routine(TEMPLATE / 'onia_legs_dummy.f', 'fks_onia_leg'))
            (work / 'sij.f').write_text('\n'.join(routines))
            (work / 'driver.f90').write_text(DRIVER)
            build = subprocess.run([
                shutil.which('gfortran'), '-O2', '-std=legacy', '-fcheck=all',
                '-ffixed-line-length-none', '-ffree-line-length-none', '-I', str(work),
                str(TEMPLATE / 'fks_phase_space_data.f'),
                str(TEMPLATE / 'genps_fks_helpers.f'), str(work / 'sij.f'),
                str(ROOT / 'HELAS/boostx.F'), str(ROOT / 'HELAS/rotxxx.F'),
                str(work / 'driver.f90'), '-o', str(work / 'check')],
                cwd=work, capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            result = subprocess.run([str(work / 'check')], cwd=work,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS light-system soft S partition', result.stdout)


if __name__ == '__main__':
    unittest.main()
