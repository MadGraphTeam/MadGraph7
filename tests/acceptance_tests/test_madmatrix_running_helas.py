"""Running custom-propagator parameters, against Fortran and across event lanes."""

import array
import glob
import math
import os
import re
import subprocess
import tempfile
import unittest

from madgraph.interface.master_interface import MasterCmd
from tests.input_files.running_helas_model import create_model


class TestMadMatrixRunningHelas(unittest.TestCase):
    def test_madmatrix_running_helas_parameters(self):
        with tempfile.TemporaryDirectory(prefix='running_helas_') as directory:
            cmd = MasterCmd()
            cmd.exec_cmd('import model %s' % create_model(os.path.join(directory, 'running_ufo')),
                         errorhandling=False)
            cmd.exec_cmd('generate e+ e- > z > mu+ mu-', errorhandling=False)
            roots = {}
            for output_format in ('standalone_fortran', 'standalone'):
                root = os.path.join(directory, output_format)
                cmd.exec_cmd('output %s %s -f' % (output_format, root), errorhandling=False)
                roots[output_format] = root
            fortran = roots['standalone_fortran']
            cpp = roots['standalone']
            fdir, = glob.glob(os.path.join(fortran, 'SubProcesses', 'P*'))
            cdir, = glob.glob(os.path.join(cpp, 'SubProcesses', 'P*'))
            self.run_command(['make', '-j2'], os.path.join(fortran, 'Source'))
            self.run_command(['make', 'check'], fdir)
            reference = {}
            for alphas in (0.08, 0.16):
                self.set_alphas(fortran, alphas)
                reference[alphas] = self.matrix_element(self.run_command(['./check', '200'], fdir))
                self.assertGreater(reference[alphas], 0.)
            self.assertGreater(abs(reference[0.16] / reference[0.08] - 1.), 0.1)

            alphas = [0.08, 0.12, 0.16, 0.10] * 16
            alpha_file = os.path.join(directory, 'alphas.bin')
            self.write_array(alpha_file, alphas)
            momenta = os.path.join(directory, 'momenta.bin')
            baseline = None
            # f exercises the scalar loop used to compute SIMD float couplings;
            # v exercises distinct vertex and parameter/denominator precision.
            configurations = [('scalar', 'd'), ('simd_128', 'd'),
                              ('simd_128', 'f'), ('simd_128', 'v')]
            gpu_backend = os.environ.get('MG7_TEST_GPU_BACKEND')
            if gpu_backend:
                self.assertIn(gpu_backend, ('cuda', 'hip'))
                configurations.append((gpu_backend, 'd'))
            for backend, precision in configurations:
                self.run_command(['make', 'cleanall'], cdir)
                self.run_command(['make', '-j2', 'BACKEND=' + backend, 'FPTYPE=' + precision], cdir)
                tolerance = 2e-5 if precision == 'f' else 1e-6
                for alpha, expected in reference.items():
                    self.set_alphas(cpp, alpha)
                    actual = self.matrix_element(self.run_command(['./check_sa.exe', 'matrix', '200'], cdir))
                    self.assert_close(actual, expected, tolerance)
                mefile = os.path.join(directory, 'me.bin')
                args = ['./check_sa.exe', 'perf', '1', '64', '1', '--energy', '200',
                        '--alphas', alpha_file, '--dump-me', mefile]
                if baseline is None:
                    self.run_command(args + ['--dump-momenta', momenta], cdir)
                    baseline = self.read_array(mefile)
                    self.assertEqual(len(baseline), len(alphas))
                    fixed_file = os.path.join(directory, 'fixed.bin')
                    fixed_args = ['./check_sa.exe', 'perf', '1', '64', '1', '--energy', '200',
                                  '--alphas', '0.08', '--momenta', momenta, '--dump-me', fixed_file]
                    self.run_command(fixed_args, cdir)
                    fixed = self.read_array(fixed_file)
                    self.assertEqual(len(fixed), len(alphas))
                    # Every amplitude is multiplied by (1+i*r)/(s-MZ^2*(1+r)+i*MZ*WZ).
                    def factor(alpha):
                        r = (alpha / 0.1)**2
                        return (1 + r*r) / ((200**2 - 91.188**2*(1+r))**2
                                             + (91.188*2.441404)**2)
                    for actual, value, alpha in zip(baseline, fixed, alphas):
                        self.assert_close(actual, value * factor(alpha) / factor(0.08), 1e-6)
                else:
                    self.run_command(args + ['--momenta', momenta], cdir)
                    values = self.read_array(mefile)
                    self.assertEqual(len(values), len(baseline))
                    for actual, expected in zip(values, baseline):
                        self.assert_close(actual, expected, tolerance)

    def run_command(self, args, cwd):
        result = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, '%s\n%s' % (args, result.stdout[-12000:]))
        return result.stdout

    def matrix_element(self, output):
        found = re.search(r'Matrix element\s*=\s*([\d.eE+-]+)', output)
        self.assertIsNotNone(found, output)
        return float(found.group(1))

    def assert_close(self, actual, expected, tolerance):
        self.assertTrue(math.isfinite(actual) and expected > 0.)
        self.assertLessEqual(abs(actual / expected - 1.), tolerance,
                             '%s != %s' % (actual, expected))

    @staticmethod
    def set_alphas(root, value):
        path = os.path.join(root, 'Cards', 'param_card.dat')
        with open(path) as stream:
            card = stream.read()
        card, count = re.subn(r'(?m)^(\s*3\s+)\S+(\s*#\s*aS\b)',
                              lambda m: m[1] + str(value) + m[2], card)
        assert count == 1
        with open(path, 'w') as stream:
            stream.write(card)

    @staticmethod
    def write_array(path, values):
        with open(path, 'wb') as stream:
            array.array('d', values).tofile(stream)

    @staticmethod
    def read_array(path):
        values = array.array('d')
        with open(path, 'rb') as stream:
            values.frombytes(stream.read())
        return values
