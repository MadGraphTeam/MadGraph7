"""Check the combined FxFx/Delta integrand with independent analytic weights."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestFxFxDeltaWeights(unittest.TestCase):
    def test_real_transfer_and_native_counterterms(self):
        with tempfile.TemporaryDirectory(prefix='mg5_fxfx_delta_weights_') as tmp:
            work = Path(tmp)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\n'
                                 'parameter(nexternal=5,nincoming=2)',
                'nFKSconfigs.inc': 'integer fks_configs\nparameter(fks_configs=1)',
                'coupl.inc': 'double precision g\ncommon/test_couplings/g',
                'orders.inc': 'integer nsplitorders,amp_split_size,qcd_pos,cpower_pos\n'
                              'parameter(nsplitorders=1,amp_split_size=1,qcd_pos=1,cpower_pos=0)\n'
                              'double precision amp_split(1)\ncommon/test_amplitudes/amp_split',
                'pmass.inc': 'pmass=0d0',
                'run.inc': 'integer ickkw,ktscheme\ncommon/test_run/ickkw,ktscheme\n'
                           'double precision real_channels_x0\n'
                           'common/c_real_channels_x0/real_channels_x0',
            }
            for name, source in includes.items():
                (work / name).write_text(''.join(
                    '      ' + line + '\n' for line in source.splitlines()))
            for name in ('timing_variables.inc', 'fks_powers.inc'):
                shutil.copyfile(TEMPLATE / name, work / name)
            routines = ('compute_native_NLOPS_weights', 'compute_MC_subt_term',
                        'compute_real_emission', 'compute_soft_counter_term',
                        'compute_collinear_counter_term',
                        'compute_soft_collinear_counter_term', 'set_FxFx_scale')
            (work / 'weights.f').write_text('\n'.join(
                fortran_routine(TEMPLATE / 'fks_singular.f', name) for name in routines))
            executable = work / 'check_fxfx_delta_weights'
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-std=legacy', '-fcheck=all',
                '-ffixed-line-length-none', '-I', str(work),
                str(TEMPLATE.parent / 'Source/timing.f'),
                str(ROOT / 'tests/input_files/check_fxfx_delta_weights.f90'),
                str(work / 'weights.f'), '-o', str(executable)],
                cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS combined FxFx Delta weights', result.stdout)


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestFxFxDeltaHMinRecords(unittest.TestCase):
    def test_only_h_records_from_the_event_are_scaled(self):
        with tempfile.TemporaryDirectory(prefix='mg5_fxfx_delta_hmin_') as tmp:
            work = Path(tmp)
            (work / 'records.f').write_text(
                fortran_routine(TEMPLATE / 'driver_mintMC.f', 'scale_H_records'))
            (work / 'check.f90').write_text('''module weight_lines
  integer :: icontr=5
  logical :: H_event(5)=[.true.,.false.,.true.,.false.,.true.]
  double precision :: wgt(3,5)
end module
program check
  use weight_lines
  implicit none
  integer :: i
  do i=1,5
    wgt(:,i)=[1d0,2d0,3d0]*i
  enddo
  call scale_H_records(3,0.25d0)
  ! Records before the event and S records keep their weights.
  if (any(wgt(:,1)/=[1d0,2d0,3d0]).or.any(wgt(:,2)/=[2d0,4d0,6d0]).or. &
      any(wgt(:,4)/=[4d0,8d0,12d0])) stop 1
  if (any(wgt(:,3)/=[0.75d0,1.5d0,2.25d0]).or. &
      any(wgt(:,5)/=[1.25d0,2.5d0,3.75d0])) stop 2
  print *, 'PASS H records'
end program
''')
            executable = work / 'check'
            result = subprocess.run([
                shutil.which('gfortran'), '-std=legacy', '-fcheck=all',
                str(work / 'check.f90'), str(work / 'records.f'),
                '-o', str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS H records', result.stdout)
