"""Lane switches swap the per-point shower scales without aliasing points."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

TEMPLATE = Path(__file__).resolve().parents[3] / 'Template/NLO/SubProcesses'

STUBS = """
module process_module
  implicit none
  integer :: next_n=0,next_n1=0,ickkw_mod=0,max_flows_n=1
  logical :: qed_matching=.false.
  double precision,allocatable :: mass_n(:),mass_n1(:),charge_n(:),charge_n1(:)
  integer,allocatable :: colour_n(:),colour_n1(:),iRtoB(:),iBtoR(:),pdg_n(:),pdg_n1(:)
  logical,allocatable :: valid_dipole_n(:,:,:),valid_dipole_n1(:,:),qed_dipole_n(:,:),qed_dipole_n1(:,:)
end module
module scale_module
  implicit none
  double precision :: global_ref_scale=0d0,shower_scale_hard=0d0,showerscaleS_hard=0d0
  integer :: born_flow_picked=0
  double precision,allocatable,dimension(:,:) :: shower_scale_nbody,shower_scale_nbody_min, &
       shower_scale_nbody_max,shower_scale_n1body,showerscaleS,showerscaleH,emsca_S_hard
  double precision,allocatable,dimension(:,:,:,:) :: emsca_S,emsca_H
  integer,allocatable :: event_colour_H(:,:,:,:)
end module
module mc_counterterms
  implicit none
  type native_kernel_state
     integer :: ileg=0
  end type
contains
  subroutine park_mc_kernel(state)
    type(native_kernel_state),intent(out) :: state
  end subroutine
  subroutine activate_mc_kernel(state)
    type(native_kernel_state),intent(in) :: state
  end subroutine
end module
"""

CHECK = """
program check
  use scale_module
  use mc_shower_state
  implicit none
  type(native_shower_state),target :: a,b
  character(20) option
  call get_command_argument(1,option)
  call fill(1d0)
  call park_native_shower(a)
  if(option=='twice')call park_native_shower(b)
  if(any(a%emsca_S/=1d0).or.any(a%event_colour_H/=1))error stop 1
  ! A fresh point sees the last parked values in storage of its own.
  call materialize_native_shower()
  if(any(emsca_S/=1d0).or.any(showerscaleH/=1d0).or.any(event_colour_H/=1))error stop 2
  call fill(2d0)
  if(any(a%emsca_S/=1d0).or.any(a%shower_scale_nbody/=1d0))error stop 3
  call park_native_shower(b)
  call activate_native_shower(a)
  if(any(emsca_S/=1d0).or.any(emsca_S_hard/=1d0))error stop 4
  call fill(3d0)
  call park_native_shower(a)
  if(any(a%emsca_S/=3d0).or.any(b%emsca_S/=2d0))error stop 5
  call activate_native_shower(b)
  if(any(emsca_S/=2d0).or.any(event_colour_H/=2))error stop 6
  call park_native_shower(b)
  ! Released states leave the last parked point's values active.
  call forget_parked_shower()
  if(any(emsca_S/=2d0).or.any(shower_scale_n1body/=2d0))error stop 7
  if(any(b%emsca_S/=2d0).or.any(a%emsca_S/=3d0))error stop 8
  print *,'swapped shower scales'
contains
  subroutine fill(v)
    double precision v
    if(.not.allocated(emsca_S))then
       allocate(shower_scale_nbody(4,4),shower_scale_nbody_min(4,4),shower_scale_nbody_max(4,4), &
            shower_scale_n1body(5,5),showerscaleS(2,2),showerscaleH(2,2),emsca_S_hard(3,1), &
            emsca_S(3,1,2,2),emsca_H(3,1,2,2),event_colour_H(2,5,3,1))
    endif
    shower_scale_nbody=v; shower_scale_nbody_min=v; shower_scale_nbody_max=v
    shower_scale_n1body=v; showerscaleS=v; showerscaleH=v; emsca_S_hard=v
    emsca_S=v; emsca_H=v; event_colour_H=int(v)
  end subroutine
end program
"""


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestShowerScaleSwap(unittest.TestCase):
    def test_lane_switch_swaps_scales_without_aliasing(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work/'stubs.f90').write_text(STUBS)
            (work/'check.f90').write_text(CHECK)
            for name, text in [('nexternal.inc', 'integer nexternal,nincoming\nparameter(nexternal=5,nincoming=2)'),
                               ('born_nhel.inc', 'integer max_bcol\nparameter(max_bcol=4)'),
                               ('nFKSconfigs.inc', 'integer fks_configs\nparameter(fks_configs=3)')]:
                (work/name).write_text(''.join('      '+line+'\n' for line in text.splitlines()))
            build = subprocess.run(['gfortran', '-O2', '-fcheck=all', '-ffixed-line-length-132', '-I.',
                                    'stubs.f90', str(TEMPLATE/'mc_shower_state.f'), 'check.f90', '-o', 'check'],
                                   cwd=work, capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout+build.stderr)
            result = subprocess.run(['./check'], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('swapped shower scales', result.stdout)
            result = subprocess.run(['./check', 'twice'], cwd=work, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Parked a point with stale scales', result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
