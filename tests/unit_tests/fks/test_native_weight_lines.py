"""Park independent native records without copying or mixing their allocations."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import TEMPLATE


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestNativeWeightLines(unittest.TestCase):
    def test_distinct_contributions_and_rejection_ownership(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            (work/'check.f90').write_text(CHECK)
            subprocess.run(['gfortran', '-O3', '-fno-fast-math', '-fcheck=all', '-ffixed-line-length-132',
                            str(TEMPLATE/'mc_context_memory.f90'), str(TEMPLATE/'weight_lines.f'), 'check.f90', '-o', 'check'],
                           cwd=work, check=True, text=True, capture_output=True)
            for option in ('restore', 'unparked', 'already_parked', 'occupied'):
                with self.subTest(option=option):
                    result = subprocess.run(['./check', option], cwd=work, capture_output=True, text=True)
                    if option == 'restore':
                        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                        self.assertIn('restored both native records', result.stdout)
                    else:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('native weight lines', result.stderr.lower())

    def test_finished_stores_are_reused_at_their_capacity(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            (work/'check.f90').write_text(RECYCLE)
            subprocess.run(['gfortran', '-O3', '-fno-fast-math', '-fcheck=all', '-ffixed-line-length-132',
                            str(TEMPLATE/'mc_context_memory.f90'), str(TEMPLATE/'weight_lines.f'), 'check.f90', '-o', 'check'],
                           cwd=work, check=True, text=True, capture_output=True)
            result = subprocess.run(['./check'], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('reused finished stores', result.stdout)


RECYCLE = """
program check
  use weight_lines
  implicit none
  type(native_weight_lines) a,b
  ! A finished point leaves a grown store with contributions and the
  ! optional calibration and legacy arrays.
  call weight_lines_allocated(5,40,3,2)
  call allocate_legacy_weight_snapshots(5)
  call allocate_born_spread_lines(3)
  icontr=37; iwgt=3; icontr_picked=5; mc_H_only=.true.
  nonfinite_contribution=.true.; nonfinite_sector=2
  call park_weight_lines(a)
  ! Nothing to adopt yet; an active store is never recycled or replaced.
  if(adopt_spare_weight_lines(5))error stop 1
  call weight_lines_allocated(5,1,1,1)
  call recycle_weight_lines(a)
  if(nspare_lines/=0)error stop 2
  if(adopt_spare_weight_lines(5))error stop 3
  call park_weight_lines(b)
  call recycle_weight_lines(a)
  call recycle_weight_lines(a)
  if(nspare_lines/=1)error stop 4
  ! A store for a different leg count is not handed out.
  if(adopt_spare_weight_lines(6))error stop 5
  ! The adopted store keeps its capacity and looks like a fresh one.
  if(.not.adopt_spare_weight_lines(5))error stop 6
  if(nspare_lines/=0.or..not.allocated(itype))error stop 7
  if(max_contr/=40.or.size(itype)/=40.or.max_wgt/=3.or.max_iproc/=2)error stop 8
  if(icontr/=0.or.iwgt/=0.or.icontr_picked/=0.or.mc_H_only)error stop 9
  if(nonfinite_contribution.or.nonfinite_sector/=0)error stop 10
  if(allocated(unwgt_B).or.allocated(parton_pdf).or.allocated(icontr_sum))error stop 11
  if(born_spread_fold_capacity/=0)error stop 12
  call deallocate_weight_lines
  if(nspare_lines/=0.or.allocated(spare_lines))error stop 13
  print *, 'reused finished stores'
end program
"""

CHECK = """
program check
  use weight_lines
  implicit none
  type(native_weight_lines) a,b
  character(20) option
  call get_command_argument(1,option)
  if(option=='unparked')call activate_weight_lines(a)
  call weight_lines_allocated(5,4,3,2)
  call allocate_legacy_weight_snapshots(5)
  call allocate_born_spread_lines(3)
  rejected_nonfinite_points=1
  icontr=4
  iwgt=3
  native_ids(:,4)=[3,1,6]
  event_nfks(4)=2
  ifold_cnt(4)=2
  wgt(:,4)=[-15d0,6d0,8d0]
  parton_pdg(:,2,4)=[2,-2,11,-11,21]
  pdg(:,0)=[1,2,3,4,5]
  momenta_m(0:3,1,2,4)=[40d0,-3d0,2d0,30d0]
  parton_pdf(1,2,4)=6
  unwgt_B(2,3)=-37d0
  call park_weight_lines(a)
  if(option=='already_parked')call park_weight_lines(a)
  if(allocated(itype).or.allocated(parton_pdf).or.icontr/=0)error stop 1
  call weight_lines_allocated(5,2,1,1)
  if(option=='occupied')call activate_weight_lines(a)
  icontr=2
  ifold_cnt(2)=3
  wgt(:,2)=[1d0,2d0,3d0]
  nonfinite_contribution=.true.; nonfinite_sector=3; rejected_nonfinite_points=2
  call park_weight_lines(b)
  call activate_weight_lines(a)
  if(icontr/=4.or.iwgt/=3.or.ifold_cnt(4)/=2.or.event_owner(4)/=2)error stop 2
  if(any(wgt(:,4)/=[-15d0,6d0,8d0]).or.any(native_ids(:,4)/=[3,1,6]))error stop 3
  if(any(pdg(:,0)/=[1,2,3,4,5]).or.any(parton_pdg(:,2,4)/=[2,-2,11,-11,21]))error stop 4
  if(any(momenta_m(0:3,1,2,4)/=[40d0,-3d0,2d0,30d0]))error stop 5
  if(parton_pdf(1,2,4)/=6.or.born_spread_fold_capacity/=3.or.unwgt_B(2,3)/=-37d0)error stop 6
  if(nonfinite_contribution.or.nonfinite_sector/=0.or.rejected_nonfinite_points/=2)error stop 7
  call park_weight_lines(a)
  call activate_weight_lines(b)
  if(icontr/=2.or.ifold_cnt(2)/=3.or.any(wgt(:,2)/=[1d0,2d0,3d0]))error stop 8
  if(allocated(parton_pdf).or.allocated(unwgt_B))error stop 9
  if(.not.nonfinite_contribution.or.nonfinite_sector/=3)error stop 10
  print *, 'restored both native records'
end program
"""
