"""Polarized production lanes agree with independently framed scalar MEs."""
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from madgraph import MG5DIR
from tests.acceptance_tests import test_nlo_real_offload_contract as offload


class TestNLOFrameRouting(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='mg7_frame_routing_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_polarized_production_lanes(self):
        self._check_production_lanes()

    def test_polarized_offloaded_production_lanes(self):
        self._check_production_lanes(backend='scalar')

    def _check_production_lanes(self, backend=None):
        root = Path(MG5DIR)
        output = Path(self.tmpdir)/'process'
        cmd = offload.TestNLORealFortranOracle._new_cmd()
        for line in ('import model loop_sm',
                     'set apply_flavor_grouping False --no_save',
                     'generate u u~ > z{0} g [real=QCD]',
                     f'output standalone_fortran --fks --limits {output}'
                     + (' --me_exporter=mg7' if backend else '')):
            offload.TestNLORealFortranOracle._run(cmd, line)
        sub = next((output/'SubProcesses').glob('P*'))
        source = (root/'tests/input_files/nlo_frame_driver.f').read_text()
        # The reference always uses native Fortran, also in the offloaded test.
        source = source.replace('call smatrix_real_frame(', 'call frame_native_real(')
        native = 'smatrix_real_fortran' if backend else 'smatrix_real'
        source += f'''
      subroutine frame_native_real(p,wgt)
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      double precision p(0:3,nexternal),pf(0:3,nexternal),wgt
      double precision ret(amp_split_size)
      integer ids(nexternal),nfksprocess
      common/c_nfksprocess/nfksprocess
      call get_frame_mask_real(nfksprocess,ids)
      call boost_to_me_frame(p,nexternal,ids,pf)
      call {native}(pf,ret,wgt)
      end
'''
        (sub/'frame_pipeline.f').write_text(source)
        subprocess.check_call(['make', '-j1'], cwd=output/'Source')
        objects = [p.with_suffix('.o').name for p in sub.glob('matrix_*.f')]
        objects += [p.with_suffix('.o').name for p in sub.glob('b_sf_*.f')]
        objects += ['sborn_sf.o']
        objects += ['real_me_chooser.o', 'splitorders_stuff.o', 'born.o', 'frame_pipeline.o']
        backend_options = []
        bridge_flags = []
        if backend:
            backend_options = ['NLO_REAL_RUNTIME_BACKEND='+backend]
            subprocess.check_call(['make', '-f', 'nlo_real.mk', 'BACKEND='+backend,
                                   'FPTYPE=d', '-j1'], cwd=sub)
            objects += ['nlo_real_offload.o']
            bridge_flags = ['-L.', '-lnlo_real_bridge', '-Wl,-rpath,$ORIGIN', '-lstdc++', '-ldl']
        subprocess.check_call(['make', '-j1', *objects, *backend_options], cwd=sub)
        flags = ['gfortran', '-O0', '-I../../lib', '-I../../Source/MODEL',
                 '-ffixed-line-length-132', '-ffunction-sections', '-fdata-sections']
        for source, target in (('boost_to_frame.f', 'frame_boost.o'), ('driver_vec.f', 'frame_vec.o')):
            subprocess.check_call(flags+['-c', source, '-o', target], cwd=sub)
        (sub/'frame_lo2.f').write_text('\n'.join(
            (root/'Template/NLO/SubProcesses/ewsudakov_functions_dummy.f').read_text().splitlines()[:16])+'\n')
        subprocess.check_call(flags+['-c', 'frame_lo2.f', '-o', 'frame_lo2.o'], cwd=sub)
        subprocess.check_call(flags+['-c', str(root/'Template/Common/Source/impose_frame_rest.f'),
                                     '-o', 'frame_rest.o'], cwd=sub)
        subprocess.check_call(['gfortran', '-Wl,--gc-sections', '-o', 'frame_pipeline', *objects,
                               'driver.o', 'frame_lo2.o', 'frame_rest.o', 'frame_boost.o',
                               'frame_vec.o', '-L../../lib', '-ldhelas', '-lmodel', *bridge_flags], cwd=sub)

        def point(vectors, masses):
            energies = [math.sqrt(sum(x*x for x in v)+m*m) for v, m in zip(vectors, masses)]
            beam = sum(energies)/2
            return [[beam, 0, 0, beam], [beam, 0, 0, -beam]] + [[e, *v] for e, v in zip(energies, vectors)]

        momenta = []
        for scale in (1., 1.4):
            momenta += point([[scale*x for x in v] for v in
                              [[100,40,60], [-60,30,-20], [-40,-70,-40]]], [91.188,0,0])
            momenta += point([[scale*x for x in v] for v in
                              [[100,40,60], [-100,-40,-60]]], [91.188,0])
        data = '\n'.join(' '.join('%.17e' % x for x in v) for v in momenta)+'\n'
        # Native scalar helicity identities are cached for the run's fixed frame.
        # Each frame needs a fresh process, initialized before helicity discovery.
        result = ''.join(subprocess.check_output(['./frame_pipeline'], cwd=sub,
                          input=str(frame)+'\n'+data, text=True,
                          stderr=subprocess.STDOUT) for frame in (6, 8))
        (Path(self.tmpdir)/'frame_result.log').write_text(result)
        if backend:
            self.assertIn('offload initialized: backend='+backend, result)
            self.assertNotIn('Fortran fallback', result)
        checks = links = 0
        for line in result.splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] == 'FRAME_CHECK':
                actual, reference = map(float, fields[3:])
                self.assertTrue(math.isfinite(actual) and math.isfinite(reference))
                self.assertAlmostEqual(actual, reference, delta=abs(reference)*1e-12)
                checks += 1
            elif fields[0] in ('FRAME_ARRAYS', 'FRAME_COL'):
                for error in map(float, fields[3:]):
                    self.assertTrue(math.isfinite(error))
                    self.assertLessEqual(error, 1e-12)
            elif fields[0] == 'FRAME_LINK':
                actual, reference, error = map(float, fields[5:])
                self.assertTrue(all(math.isfinite(value) for value in (actual, reference, error)))
                self.assertAlmostEqual(actual, reference, delta=abs(reference)*1e-12)
                self.assertLessEqual(error, 1e-12)
                links += 1
        self.assertEqual(checks, 8)
        self.assertEqual(links, 12)
        self._check_isr_phase(sub, flags, data)

    def _check_isr_phase(self, sub, flags, data):
        """Contract nonzero spin amplitudes with the actual scalar/store phases."""
        root = Path(MG5DIR)
        source = (root/'Template/NLO/SubProcesses/fks_singular.f').read_text()
        routines = []
        for name in ('sborncol_isr', 'sborncol_isr_store', 'AP_reduced',
                     'Qterms_reduced_spacelike', 'getaziangles'):
            start = source.index('      subroutine '+name+'(')
            routines.append(source[start:source.index('\n      end\n', start)+11])
        (sub/'isr_blocks.f').write_text('\n'.join(routines))
        shutil.copy2(root/'tests/input_files/nlo_isr_frame_driver.f', sub/'isr_frame.f')
        for name in ('isr_blocks', 'isr_frame'):
            subprocess.check_call(flags+['-c', name+'.f', '-o', name+'.o'], cwd=sub)
        subprocess.check_call(['gfortran', '-Wl,--gc-sections', '-o', 'isr_frame',
                               'isr_blocks.o', 'isr_frame.o', 'frame_boost.o', 'frame_rest.o',
                               '-L../../lib', '-lgeneric', '-ldhelas', '-lmodel'], cwd=sub)
        probe = subprocess.run(['./isr_frame'], cwd=sub, input=data, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        result = probe.stdout
        (Path(self.tmpdir)/'isr_result.log').write_text(result)
        self.assertEqual(probe.returncode, 0, result)
        self.assertEqual(result.count('ISR_FRAME'), 16)
        for line in result.splitlines():
            if line.split()[:1] == ['ISR_FRAME']:
                values = list(map(float, line.split()[5:]))
                self.assertTrue(all(math.isfinite(value) for value in values))
                self.assertLessEqual(values[-1], 1e-12)
