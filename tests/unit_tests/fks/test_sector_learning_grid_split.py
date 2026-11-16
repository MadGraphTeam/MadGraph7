"""MINT0 grid accumulation of ordinary points with learned sector maps."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_born_spreading import mint_support, TEMPLATE


PROGRAM = r'''
program exercise_grid_split
use mint_module
implicit none
character(32) :: action
integer :: pdgs(2,5),n,kdim
double precision :: xin(ndimmax),xout(ndimmax),ratio,born,sector1,sector2
call get_command_argument(1,action)
call DefaultFKSParam
sector_learning_active=.true.;fixed_order=.false.;born_spread_active=.false.
imode=0;n_ord_virt=0;ndim=5;nchans=1;ichan=1;nint_used=4;ifold=1
do kdim=1,ndim
 xgrid(0:4,kdim,1)=[(dble(n)/4d0,n=0,4)]
enddo
xacc=0d0;nhits=0;pdgs=21
call sector_learning_register_layout(1,[1,2],[1,2],pdgs)
! Sector 1 keeps the sampled xi/y cells; sector 2 moves by (1/4,2/4).
unified_slots(1)%qcodes(:,1)=[0,0]
unified_slots(1)%qcodes(:,2)=[1,2]
if(trim(action).eq.'noshift')unified_slots(1)%qcodes(:,2)=[0,0]
if(trim(action).eq.'probe')sector_learning_collect=.true.
if(trim(action).eq.'mint1')then
 imode=1;sl_ready=.true.
endif
call sector_learning_begin_point()
call sector_learning_select_slot(1)
xin=0d0;xin(1:ndim)=.1d0;icell=0;icell(1:ndim)=1
call resample_sector_radiation(xin,xout,ratio,1)
if(any(xout.ne.xin).or.ratio.ne.1d0)stop 501
call resample_sector_radiation(xin,xout,ratio,2)
if(trim(action).ne.'noshift')then
 if(abs(xout(ndim-2)-.35d0).gt.1d-15.or.abs(xout(ndim-1)-.6d0).gt.1d-15.or.ratio.ne.1d0)stop 502
endif
! Contribution 1 is the Born part; 2-3 belong to sector 1, 4-5 to sector 2.
call sector_learning_mark_range(2,3,1)
call sector_learning_mark_range(4,5,2)
call sector_learning_split_add(1,1,3d0)
call sector_learning_split_add(2,1,-1d0)
call sector_learning_split_add(4,1,2d0)
call sector_learning_split_add(1,2,-1d0)
call sector_learning_split_add(4,2,-1d0)
call sector_learning_split_add_h(3,.5d0)
call sector_learning_split_add_h(5,.25d0)
! S events |3-1+2|=4 and |-1-1|=2, H events .5 and .25.
f=0d0;f(1)=6.75d0
if(trim(action).eq.'mismatch')f(1)=7d0
call sector_learning_split_close(f(1))
if(trim(action).eq.'reject')call sector_learning_reject_point()
call add_point_to_grids(xin)
do kdim=1,ndim
 if(abs(sum(xacc(1:4,kdim,1))-f(1)).gt.1d-12)stop 503
enddo
if(any(xacc(2:4,ndim,1).ne.0d0).or.xacc(1,ndim,1).ne.f(1))stop 504
if(trim(action).eq.'split')then
 ! Each S event is split in proportion to |B| and the |R_s|: 4 as 3:1:2, 2 as 1:0:1.
 born=4d0*3d0/6d0+1d0
 sector1=4d0*1d0/6d0+.5d0
 sector2=4d0*2d0/6d0+1d0+.25d0
 if(abs(xacc(1,ndim-2,1)-born-sector1).gt.1d-12.or.abs(xacc(2,ndim-2,1)-sector2).gt.1d-12)stop 505
 if(abs(xacc(1,ndim-1,1)-born-sector1).gt.1d-12.or.abs(xacc(3,ndim-1,1)-sector2).gt.1d-12)stop 506
 print *,'PASS Born and sector shares in the cells where they were evaluated'
else
 ! All of the ABS weight in the sampled cells, as without learned maps.
 if(any(abs(xacc(1,1:ndim,1)-f(1)).gt.1d-12).or.any(xacc(2:4,1:ndim,1).ne.0d0))stop 507
 print *,'PASS ',trim(action),' accumulates in the sampled cells'
endif
end program
double precision function ran2()
call random_number(ran2)
end function
'''


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestSectorLearningGridSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='mg7-grid-split-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        support = mint_support(cls.work, real_fks_params=True)
        exposed = cls.work / 'exposed_mint.f90'
        exposed.write_text((TEMPLATE / 'mint_module.f90').read_text().replace('private', 'public'))
        support = [str(exposed) if path == str(TEMPLATE / 'mint_module.f90') else path for path in support]
        program = cls.work / 'check.f90'
        program.write_text(PROGRAM)
        cls.executable = cls.work / 'check'
        result = subprocess.run(['gfortran', '-O0', '-fcheck=all', '-ffree-line-length-none', '-ffixed-line-length-132',
                                 '-I', str(cls.work), *support, str(program), '-o', str(cls.executable)],
                                cwd=cls.work, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def run_case(self, action):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([str(self.executable), action], cwd=directory,
                                    capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn('PASS', output)
        return output

    def test_shares_go_to_the_cells_where_they_were_evaluated(self):
        self.assertNotIn('Grid split differs', self.run_case('split'))

    def test_unshifted_sectors_match_the_sampled_cell_accumulation(self):
        self.run_case('noshift')

    def test_inconsistent_split_falls_back_to_the_sampled_cells(self):
        self.assertIn('Grid split differs from the ABS weight', self.run_case('mismatch'))

    def test_rejected_point_falls_back_to_the_sampled_cells(self):
        self.run_case('reject')

    def test_probes_and_mint1_do_not_split(self):
        self.run_case('probe')
        self.run_case('mint1')


if __name__ == '__main__':
    unittest.main()
