"""Capture routed S scalars before ABS for the tiled S-event maps (FKSSectorLearning).

The probe capture must preserve folds, canonical flavours and outer-sector owners."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_born_spreading import TEMPLATE, mint_support, routine
from tests.unit_tests.fks import test_born_flow_rejection as driver_fixture

PROGRAM = r'''
program routed_capture
 use mint_module,fixture_mint_k=>k
 use weight_lines
 implicit none
 integer :: pdg_type_d(3,5),fks_i_d(3)
 common/test_fks_info/pdg_type_d,fks_i_d
 integer :: iproc_save(3),eto(3,3),etoi(3,3),maxproc_found
 common/cproc_combination/iproc_save,eto,etoi,maxproc_found
 integer :: fold,s,k,t,i,j,first,last,owner,ids(3)=[3,1,2],providers(3)=[2,1,2],pdgs(3,5)
 double precision :: value(3),expected_b(2,16),expected_r(2,16,3),loss(1),b(2,16,1),r(2,16,3,1)
 integer :: perms(16,3)
 FKSSectorLearning=.true.
 pdgs=21
 call sector_learning_register_layout(1,ids,providers,pdgs)
 sector_learning_collect=.true.
 call sector_learning_begin_point
 call sector_learning_select_slot(1)
 ndim=5;ifold=1;imode=1;only_virt=.false.
 call weight_lines_allocated(5,256,1,3)
 icontr=0;expected_b=0d0;expected_r=0d0
 pdg_type_d=21;fks_i_d=5;iproc_save=3;maxproc_found=3
 eto(:,1)=[2,1,2];eto(:,2)=[1,2,1];eto(:,3)=[2,2,1];etoi=0
 do fold=1,16
   ! Node-dependent Born, integrated, average-virtual and excluded residual.
   do t=1,4
     value=[dble(fold+t),-dble(2*fold+t),dble(3*t)]/16d0
     call record(merge(14,merge(15,t+1,t.eq.3),t.eq.4),2,.false.,value)
     if(t.ne.4)call accumulate(expected_b(:,fold),2,value)
   enddo
   do s=1,3
     first=icontr+1
     ! Native sector identity differs from the OUTER owner. Capture follows
     ! existing eto routing but associates both S terms with that owner.
     k=mod(ids(s),3)+1
     value=[dble(fold+2*s),-dble(fold+s),dble(s)]/16d0
     call record(4,k,.false.,value)
     call accumulate(expected_r(:,fold,s),k,value)
     value=[-dble(3*fold+s),dble(fold),-dble(s)]/32d0
     call record(5,k,.false.,value)
     call accumulate(expected_r(:,fold,s),k,value)
     value=999d0
     call record(1,k,.true.,value)
     call sector_learning_mark_range(first,icontr,ids(s))
   enddo
 enddo
 ! Exercise tag growth beyond64 and flavour-buffer growth from1 to2.
 if(size(sl_owners).lt.icontr)stop 1
 call sum_identical_contributions
 if(maxval(abs(sl_point_b(1:2,:)-expected_b)).gt.1d-12)stop 2
 if(maxval(abs(sl_point_r(1:2,:,:)-expected_r)).gt.1d-12)stop 3
 if(sl_point_flavours.ne.2)stop 4
 b(:,:,1)=sl_point_b(1:2,:);r(:,:,:,1)=sl_point_r(1:2,:,:)
 ! Identity, a (1,2) tile translation and a reversal of the 16 tiles.
 do t=1,16
   perms(t,1)=t
   perms(t,2)=1+4*mod((t-1)/4+1,4)+mod(mod(t-1,4)+2,4)
   perms(t,3)=17-t
 enddo
 call sl_tile_losses_unchecked(b,r,perms,loss)
 open(13,file='perms.dat',status='replace')
 write(13,*)perms
 close(13)
 open(12,file='capture.dat',status='replace')
 do fold=1,16
   do j=1,2
     write(12,*)fold,j,b(j,fold,1),r(j,fold,:,1)
   enddo
 enddo
 close(12)
 write(*,'(a,es26.17)')'LOSS ',loss(1)
 ! Rejection poisons the complete orbit; a new point resets all tags and
 ! buffers, including prior folds and enlarged canonical-flavour storage.
 call sector_learning_reject_point
 if(.not.sl_point_rejected)stop 5
 call sector_learning_begin_point
 call sector_learning_select_slot(1)
 if(sl_point_rejected.or.any(sl_point_b.ne.0d0).or.any(sl_point_r.ne.0d0).or.any(sl_owners.ne.0))stop 6
 call sector_learning_add_contribution(1,16,4,-7d0)
 if(sl_point_b(4,16).ne.-7d0.or.any(sl_point_r.ne.0d0).or.sl_point_flavours.ne.4)stop 7
 write(*,*)'PASS routed capture'
 contains
 subroutine accumulate(target,sector,values)
   real(8) :: target(2),values(3)
   integer :: sector,l
   do l=1,3
     target(eto(l,sector))=target(eto(l,sector))+values(l)
   enddo
 end subroutine
 subroutine record(kind,sector,is_h,values)
   integer :: kind,sector
   logical :: is_h
   real(8) :: values(3)
   icontr=icontr+1
   itype(icontr)=kind;nFKS(icontr)=sector;niproc(icontr)=3;ifold_cnt(icontr)=fold
   H_event(icontr)=is_h;parton_iproc(1:3,icontr)=values
   event_nFKS(icontr)=sector;parton_pdg(:,:,icontr)=21;momenta(:,:,icontr)=0d0
 end subroutine
end program
real(8) function ran2()
 error stop 'capture introduced an RNG draw'
end function
'''


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestSectorLearningCapture(unittest.TestCase):
    def test_actual_S_route_matches_independent_flavour_fold_loss(self):
        with tempfile.TemporaryDirectory(prefix='mg5_sector_capture_') as temp:
            work=Path(temp)
            support=mint_support(work,real_weight_lines=True)
            # Expose buffers only in this temporary test compilation. The
            # route and buffer implementations remain production source.
            source=(TEMPLATE/'mint_module.f90').read_text()
            source=re.sub(r'(?i),\s*private\b','',source)
            source=re.sub(r'(?im)^\s*private\s*$','  public',source)
            (work/'mint_public.f90').write_text(source)
            support=[str(work/'mint_public.f90') if p==str(TEMPLATE/'mint_module.f90') else p for p in support]
            inc={
                'nexternal.inc':'integer nexternal,nincoming\nparameter(nexternal=5,nincoming=2)',
                'genps.inc':'integer maxproc\nparameter(maxproc=3)',
                'nFKSconfigs.inc':'integer fks_configs\nparameter(fks_configs=3)',
                'fks_info.inc':'integer pdg_type_d(3,5),fks_i_d(3)\ncommon/test_fks_info/pdg_type_d,fks_i_d'}
            for name,text in inc.items():
                (work/name).write_text(''.join('      '+line+'\n' for line in text.splitlines()))
            (work/'context.f90').write_text('module process_module\ninteger :: ndelH=1\ninteger :: event_colour_H(2,5,3,16)=0\nend module\nmodule scale_module\nreal(8) :: emsca_H(3,16,1,1)=1d0\nend module\n')
            (work/'route.f').write_text(''.join(routine(TEMPLATE/'fks_singular.f',name) for name in ('sum_identical_contributions','pdg_equal','momenta_equal')))
            (work/'program.f90').write_text(PROGRAM)
            result=subprocess.run([shutil.which('gfortran'),'-O2','-fcheck=all','-ffpe-trap=invalid,zero,overflow','-ffixed-line-length-none',*support,str(work/'context.f90'),str(TEMPLATE/'weight_lines.f'),str(work/'route.f'),str(work/'program.f90'),'-o',str(work/'capture')],cwd=work,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(work/'capture')],cwd=work,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            b=[[0.,0.] for _ in range(16)];r=[[[0.,0.] for _ in range(16)] for _ in range(3)]
            for line in (work/'capture.dat').read_text().splitlines():
                row=line.split()
                fold,flavour=int(row[0])-1,int(row[1])-1;b[fold][flavour]=float(row[2])
                for s in range(3):r[s][fold][flavour]=float(row[3+s])
            # Fortran wrote perms(16,3) column-major: sector-major here.
            values=[int(x)-1 for x in (work/'perms.dat').read_text().split()]
            perms=[values[16*s:16*(s+1)] for s in range(3)]
            for perm in perms:
                self.assertEqual(sorted(perm),list(range(16)))
            # B stays at its node; each sector's R comes from its mapped tile.
            # Negative parts are taken per canonical flavour and node.
            expected=sum(max(-(b[node][f]+sum(r[s][perms[s][node]][f] for s in range(3))),0.)
                         for node in range(16) for f in range(2))
            self.assertGreater(expected,0.)
            actual=float(re.search(r'LOSS\s+([^\n]+)',result.stdout).group(1))
            self.assertAlmostEqual(actual,expected,places=12)

    def test_actual_driver_registers_selects_tags_and_freezes_proposal_fit(self):
        driver_fixture.TestBornFlowRejection.setUpClass()
        self.addCleanup(driver_fixture.TestBornFlowRejection.doClassCleanups)
        for mode in ('learning','learning_reject'):
            with self.subTest(mode=mode):
                result=subprocess.run([str(driver_fixture.TestBornFlowRejection.executable),mode],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('PASS '+mode,result.stdout)

    def test_actual_registration_helper_uses_ordered_outer_metadata_and_strict_provider_bounds(self):
        with tempfile.TemporaryDirectory(prefix='mg5_sector_layout_') as temp:
            work=Path(temp)
            includes={
                'nexternal.inc':'integer nexternal,nincoming\nparameter(nexternal=5,nincoming=2)',
                'nFKSconfigs.inc':'integer fks_configs\nparameter(fks_configs=4)',
                'fks_info.inc':'integer pdg_type_d(4,5)\ncommon/layout_info/pdg_type_d',
                'real_channels.inc':'integer real_sector_me(3)\nparameter(real_sector_me=[7,8,2])'}
            for name,text in includes.items():
                (work/name).write_text(''.join('      '+line+'\n' for line in text.splitlines()))
            (work/'helper.f').write_text(routine(TEMPLATE/'driver_mintMC.f','register_sector_learning_layout'))
            (work/'context.f90').write_text("""module mint_module
contains
subroutine sector_learning_register_layout(slot,ids,providers,pdgs)
 integer :: slot,ids(:),providers(:),pdgs(:,:)
 if(slot.ne.2.or.any(ids.ne.[3,1]).or.any(providers.ne.[2,7]))stop 1
 if(any(pdgs(1,:).ne.3).or.any(pdgs(2,:).ne.1))stop 2
 write(*,*)'PASS actual layout registration'
end subroutine
end module
program check_helper
 use mint_module
 integer :: pdg_type_d(4,5),ids(2)=[3,1],mode
 common/layout_info/pdg_type_d
 character(20) :: arg
 call get_command_argument(1,arg);read(arg,*)mode
 do mode=1,4
 pdg_type_d(mode,:)=mode
 enddo
 read(arg,*)mode
 if(mode.ne.0)ids(2)=mode
 call register_sector_learning_layout(2,2,ids)
end program
""")
            result=subprocess.run([shutil.which('gfortran'),'-O2','-fcheck=all','-ffixed-line-length-none',str(work/'context.f90'),str(work/'helper.f'),'-o',str(work/'layout')],cwd=work,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(work/'layout'),'0'],cwd=work,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('PASS actual layout registration',result.stdout)
            for invalid in (-1,4,5):
                with self.subTest(invalid=invalid):
                    result=subprocess.run([str(work/'layout'),str(invalid)],cwd=work,capture_output=True,text=True)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('Missing exact sector/provider metadata for learning',result.stdout)
