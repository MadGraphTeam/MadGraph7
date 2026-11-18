"""A virtual cache belongs to one complete point, including skipped folds."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import TEMPLATE, fortran_routine


def production_fixture():
    """Exercise production link collection, selection, residual and cache code."""
    source = fortran_routine(TEMPLATE / 'fks_singular.f', 'bornsoftvirtual')
    reset = source[source.index('c The first radiation trial'):source.index('      if (firsttime) then')]
    links = source[source.index(' 547  continue'):source.index('      use_surrogate=VirtTrickSurrogate.ge.1')]
    start = source.index('c The scalar fit is valid only for one virtual Born order normalized by')
    block = source[start:source.index('c eq.(MadFKS.C.13)', start)]
    driver = fortran_routine(TEMPLATE / 'driver_mintFO.f', 'sigint')
    begin_point = driver[driver.index('      call vs_begin_point'):driver.index('      if (firsttime) then')]
    return '''      subroutine evaluate_fold
      use point_fixture
      implicit none
      integer iamp,iord,i,j,k,m,n
      double precision wgt1,wgt,contr,eikIreg
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
''' + reset + links + block + '''      end
      subroutine begin_fixed_order_point
      use virt_surrogate
''' + begin_point + '''      end
''' + fortran_routine(TEMPLATE / 'fks_singular.f', 'reset_folded_virtual')


CONTEXT = '''module mint_module
integer :: ichan=1,VirtTrickSurrogate=0,born_spread_phase=0
logical :: use_poly_virtual=.false.
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
logical :: collect_surrogate,need_color_links_used=.true.,need_charge_links_used=.false.
logical :: need_color_links,need_charge_links
integer,parameter :: nsplitorders=1,qcd_pos=1,qed_pos=2,nexternal=5,i_fks=5
integer :: nFKSprocess=1,nFKSprocess_save,nFKSprocess_col=1,nFKSprocess_chg=0
integer :: fks_j_from_i(nexternal,0:nexternal)=0
real(8),parameter :: ZERO=0d0,oneo8pi2=1d0
real(8) :: pmass(nexternal)=1d0,p(0:3,nexternal)=0d0,xicut_used=1d0
real(8) :: bsv_wgt=0d0,amp_split_bsv(1)=0d0,amp_split_soft(1)=0d0,link_weight=2d0
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
p_born(:,1)=[5d0,0d0,0d0,5d0]
p_born(:,2)=[5d0,0d0,0d0,-5d0]
p_born(:,3)=[5d0,3d0,0d0,4d0]
p_born(:,4)=[5d0,-3d0,0d0,-4d0]
fks_j_from_i(i_fks,0:1)=[1,3]
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
subroutine fks_inc_chooser
end subroutine
subroutine sborn_frame(p,wgt)
real(8) p(0:3,4),wgt
wgt=1d0
end subroutine
subroutine sborn_sf_frame(p,m,n,wgt)
use point_fixture,only: link_weight
integer m,n
real(8) p(0:3,4),wgt
wgt=link_weight
end subroutine
subroutine eikonal_Ireg(p,m,n,xicut,eik)
integer m,n
real(8) p(0:3,5),xicut,eik
eik=0d0
end subroutine
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
class TestFoldingVirtual(unittest.TestCase):
    def test_delayed_endpoint_resets_previous_virtual_and_draws_once(self):
        program = '''program check
use point_fixture
implicit none
! Establish a nonzero cached result from a previous Born point.
call new_point(2d0,7d0)
fold=0
call evaluate_fold
call expect(5d0,2d0,1,1)
fold=1
call evaluate_fold
call expect(5d0,2d0,1,1)
! The new point's first radiation fold had no endpoint and never called
! bornsoftvirtual. Its first supported endpoint reaches this block with
! fold=1, exactly as for the massive shifted-fold production-map fixture.
call new_point(3d0,13d0)
fold=1
call evaluate_fold
call expect(10d0,3d0,2,2)
call evaluate_fold
call expect(10d0,3d0,2,2)
if(abs(virtual_over_born-13d0/3d0)>1d-12)error stop 'delayed Born ratio not initialized'
! A point where the stochastic virtual is skipped must also reset the
! cache, rather than reuse the preceding point's nonzero residual.
virtual_fraction=0.1d0
call new_point(4d0,100d0)
fold=1
call evaluate_fold
call expect(0d0,4d0,2,3)
call evaluate_fold
call expect(0d0,4d0,2,3)
! Calibration records no residual, and must consume no virtual draw.
born_spread_phase=1
call new_point(5d0,200d0)
call evaluate_fold
call expect(0d0,5d0,2,3)
! A later supported fold must initialize the fit's Born and fraction.
born_spread_phase=0
VirtTrickSurrogate=1
virtual_fraction=0.5d0
call new_point(6d0,19d0)
! Other contributions can have populated links before this first endpoint.
! The production block must clear them and collect this Born's links.
call vs_add_link(p_born,QES2,i_fks,3,3,100d0)
call evaluate_fold
call expect(26d0,6d0,3,4)
if(.not.vs_vcomp.or.vs_born.ne.6d0.or.vs_fvirt.ne.0.5d0.or.vs_virt.ne.19d0) &
  error stop 'surrogate state not initialized on delayed endpoint'
if(.not.vs_featok.or.vs_feat(4).ne.link_weight)error stop 'delayed endpoint colour links missing or stale'
call evaluate_fold
call expect(26d0,6d0,3,4)
call vs_features(p_born,born_wgt,QES2,VirtTrickSurrogate)
if(.not.vs_featok.or.vs_feat(4).ne.link_weight)error stop 'links changed across folds'
! Fixed-order callers own their reset just like the MC driver.
VirtTrickSurrogate=0
virtual_fraction=1d0
fold=0
call begin_fixed_order_point
born_wgt=2d0
amp_split_born_for_virt=2d0
exact_virtual=29d0
call evaluate_fold
call expect(27d0,2d0,4,5)
exact_virtual=31d0
call evaluate_fold
call expect(27d0,2d0,4,5)
call begin_fixed_order_point
call evaluate_fold
call expect(29d0,2d0,5,6)
write(*,*) 'PASS folded virtual cache'
end program
'''
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
            self.assertIn('PASS folded virtual cache', result.stdout)
