"""Write the strong coupling of the event kinematics as AQCDUP of MC@NLO events."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestEventAlphas(unittest.TestCase):
    def test_event_scale_contribution_and_bogus_soft_born(self):
        with tempfile.TemporaryDirectory(prefix='mg5_event_alphas_') as tmp:
            work = Path(tmp)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\n'
                                 'parameter(nexternal=5,nincoming=2)',
                'fks_info.inc': 'integer pdg_type_d(2,5),fks_i_d(2)\n'
                                'common/test_fks_info/pdg_type_d,fks_i_d',
                'orders.inc': 'integer nsplitorders,qcd_pos,qed_pos\n'
                              'parameter(nsplitorders=2,qcd_pos=1,qed_pos=2)',
            }
            for name, text in includes.items():
                (work / name).write_text(''.join('      '+s+'\n' for s in text.splitlines()))
            (work / 'routines.f').write_text(
                fortran_routine(TEMPLATE / 'driver_mintMC.f', 'special_check_SoftSing'))
            executable = work / 'check_event_alphas'
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-std=legacy', '-fcheck=all',
                '-ffixed-line-length-none', '-I', str(work),
                str(TEMPLATE.parent / 'Source/extra_weights.f'),
                str(TEMPLATE / 'weight_lines.f'),
                str(ROOT / 'tests/input_files/check_event_alphas.f90'),
                str(work / 'routines.f'),
                '-o', str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS event alphas', result.stdout)
