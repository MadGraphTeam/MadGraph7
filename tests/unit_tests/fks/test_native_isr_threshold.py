"""Native ISR histories at the one-particle Born mass threshold."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import (
    ROOT, TEMPLATE, fks_test_module, fortran_routine)


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestNativeISRThreshold(unittest.TestCase):
    def test_boosted_light_system_keeps_native_counterevents(self):
        with tempfile.TemporaryDirectory(prefix='mg7_native_isr_threshold_') as tmp:
            work = Path(tmp)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\n'
                                 'parameter(nexternal=4,nincoming=2)',
                'genps.inc': 'integer max_branch,max_particles\n'
                             'parameter(max_branch=8,max_particles=8)',
                'orders.inc': 'integer nsplitorders\nparameter(nsplitorders=1)',
                'run.inc': 'double precision ebeam(2),xbk(2)\ninteger lpp(2)\n'
                           'common/test_run/ebeam,xbk,lpp',
                'coupl.inc': '',
                'pmass.inc': 'common/to_mass/pmass',
            }
            for name, source in includes.items():
                (work / name).write_text(''.join('      ' + line + '\n'
                                                 for line in source.splitlines()))
            for name in ('fks_powers.inc', 'timing_variables.inc'):
                shutil.copyfile(TEMPLATE / name, work / name)
            (work / 'native_context.f90').write_text(
                'module mc_native_context\n'
                'logical :: native_mapping=.false.\nend module\n')
            (work / 'phase_space.f').write_text(fks_test_module())
            routines = [fortran_routine(TEMPLATE / 'fks_singular.f', name)
                        for name in ('rotate_invar', 'trp_rotate_invar',
                                     'phspncheck_nocms', 'xlen4', 'xmom_compare',
                                     'xmcompare', 'xprintout')]
            routines.extend(fortran_routine(ROOT / 'Template/NLO/Source/kin_functions.f', name)
                            for name in ('dot', 'rho', 'threedot'))
            (work / 'support.f').write_text('\n'.join(routines))
            executable = work / 'check_threshold'
            build = subprocess.run([
                shutil.which('gfortran'), '-O2', '-std=legacy', '-fcheck=all',
                '-ffixed-line-length-none', '-ffunction-sections', '-fdata-sections',
                '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections',
                '-I', str(work), str(TEMPLATE.parent / 'Source/timing.f'),
                str(TEMPLATE / 'fks_phase_space_data.f'),
                str(TEMPLATE / 'genps_fks_helpers.f'), str(work / 'native_context.f90'),
                str(TEMPLATE / 'FKSParams.f90'),
                str(TEMPLATE / 'genps_fks_radiation.f'), str(work / 'phase_space.f'),
                str(work / 'support.f'), str(TEMPLATE / 'boostwdir2.f'),
                str(TEMPLATE / 'resonance_recoil.f'), str(TEMPLATE / 'initial_recoil.f'),
                str(ROOT / 'HELAS/boostx.F'), str(ROOT / 'HELAS/rotxxx.F'),
                str(ROOT / 'tests/input_files/check_native_isr_threshold.f90'),
                '-o', str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            result = subprocess.run([str(executable)], cwd=work,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS native ISR one-particle threshold', result.stdout)


if __name__ == '__main__':
    unittest.main()
