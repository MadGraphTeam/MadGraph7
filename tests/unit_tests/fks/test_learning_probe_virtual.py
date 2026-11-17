"""Learning probes skip the exact virtual; test_folding_virtual checks the folded cache."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import TEMPLATE, fortran_routine


def production_fixture():
    """Exercise the production selection, residual and cache code verbatim."""
    source = fortran_routine(TEMPLATE / 'fks_singular.f', 'bornsoftvirtual')
    reset = source[source.index('c The first radiation trial'):source.index('      if (firsttime) then')]
    start = source.index('c The scalar fit is valid only for one virtual Born order normalized by')
    block = source[start:source.index('c eq.(MadFKS.C.13)', start)]
    return '''      subroutine evaluate_fold
      use point_fixture
      implicit none
      integer iamp
      logical first_virtual_evaluation,virtual_fold_ready
      common /c_folded_virtual/ virtual_fold_ready
      double precision virt_wgt_save,amp_split_virt_save(amp_split_size)
      save virt_wgt_save,amp_split_virt_save
      double precision ran2
      external ran2
      use_surrogate=VirtTrickSurrogate.ge.1
      virt_wgt=0d0
      amp_split_virt=0d0
      amp_split_avv=0d0
''' + reset + block + '''      end
''' + fortran_routine(TEMPLATE / 'fks_singular.f', 'reset_folded_virtual')


CONTEXT = '''module mint_module
integer :: ichan=1,VirtTrickSurrogate=0,born_spread_phase=0
logical :: use_poly_virtual=.false.
type evaluation_mode
logical :: integration=.true.,born_spread_samples=.false.
end type
type(evaluation_mode) :: point_evaluation
real(8) :: polyfit(0:1)=0d0,average_virtual(0:1,1)=1d0,virtual_fraction(1)=1d0
end module
module point_fixture
use virt_surrogate
use mint_module
implicit none
integer,parameter :: amp_split_size=1,timing_virtual=1
integer :: fold,ifold_counter,calls=0,draws=0
common /cfl/fold,ifold_counter
logical :: use_surrogate,timing_sampled=.false.
character(len=4) :: abrv='all '
real(8) :: p_born(0:3,4),QES2=100d0,born_wgt,virt_wgt,virtual_over_born,avv_wgt
real(8) :: amp_split_born_for_virt(1),amp_split_finite_ML(1)
real(8) :: amp_split_virt(1),amp_split_avv(1),ave_ratio(0:1)
real(8) :: exact_virtual=0d0,next_draw=0.25d0
contains
subroutine new_point(born,exact)
real(8) born,exact
call reset_folded_virtual
call vs_begin_point
born_wgt=born
amp_split_born_for_virt=born
exact_virtual=exact
virtual_over_born=0d0
end subroutine
subroutine expect(residual,approximated,ncalls,ndraws)
real(8) residual,approximated
integer ncalls,ndraws
if(abs(virt_wgt-residual)>1d-12.or.any(abs(amp_split_virt-residual)>1d-12)) &
  error stop 'stale or rescaled residual'
if(abs(avv_wgt-approximated)>1d-12.or.any(abs(amp_split_avv-approximated)>1d-12)) &
  error stop 'incorrect approximation'
if(calls.ne.ncalls.or.draws.ne.ndraws)error stop 'virtual redrawn within folded point'
end subroutine
end module
subroutine binothlha_frame(p,born,result)
use point_fixture,only: calls,exact_virtual,amp_split_finite_ML
real(8) p(0:3,4),born,result
calls=calls+1
result=exact_virtual
amp_split_finite_ML=exact_virtual
end subroutine
real(8) function ran2()
use point_fixture,only: draws,next_draw
draws=draws+1
ran2=next_draw
end function
subroutine timing_enter(label)
integer label
end subroutine
subroutine timing_leave(label)
integer label
end subroutine
'''


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestLearningProbeVirtual(unittest.TestCase):
    def run_program(self, program, marker):
        """Compile one check program with the production block and run it."""
        with tempfile.TemporaryDirectory(prefix='mg5_folded_virtual_') as tmp:
            work = Path(tmp)
            (work / 'context.f90').write_text(CONTEXT)
            (work / 'cache.f').write_text(production_fixture())
            (work / 'check.f90').write_text(program)
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all',
                '-ffpe-trap=invalid,zero,overflow', '-ffixed-line-length-none',
                str(TEMPLATE / 'virtual_surrogate.f90'), str(work / 'context.f90'),
                str(work / 'cache.f'), str(work / 'check.f90'), '-o', str(work / 'check')],
                cwd=work, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(work / 'check')], cwd=work,
                                    text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(marker, result.stdout)

    def test_learning_probe_skips_the_exact_virtual(self):
        self.run_program('''program check
use point_fixture
implicit none
! A probe records no one-loop residual (type 14): it neither draws the
! stochastic virtual selection nor evaluates the virtual, at any fold.
point_evaluation%integration=.false.
call new_point(5d0,200d0)
fold=0
call evaluate_fold
call expect(0d0,5d0,0,0)
fold=1
call evaluate_fold
call expect(0d0,5d0,0,0)
! Without the exact virtual the surrogate fit returns at once.
if(vs_vcomp)error stop 'probe marked the exact virtual as computed'
! The same point outside a probe draws once and evaluates the virtual.
point_evaluation%integration=.true.
call new_point(5d0,200d0)
fold=0
call evaluate_fold
call expect(195d0,5d0,1,1)
write(*,*) 'PASS probe skips the virtual'
end program
''', 'PASS probe skips the virtual')
