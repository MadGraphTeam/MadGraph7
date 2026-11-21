"""Numerical regressions for compensated binary64 propagator invariants."""
from decimal import Decimal, localcontext
from pathlib import Path
import math
import platform
import random
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('g++'), 'requires a C++ compiler')
class TestCompensatedInvariant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        directory = Path(cls.directory.name)
        source = directory / 'probe.cc'
        source.write_text('''#include "Arithmetics/Invariant.h"
#include <iostream>
#include <iomanip>
int main() {
  double a[4], b[4], m;
  while(std::cin >> a[0] >> a[1] >> a[2] >> a[3]
                 >> b[0] >> b[1] >> b[2] >> b[3] >> m) {
    double q[4];
    for(int k=0;k<4;++k) q[k]=a[k]+b[k];
    const double naive=q[0]*q[0]-q[1]*q[1]-q[2]*q[2]-q[3]*q[3]-m*m;
    std::cout << std::setprecision(17)
              << MG_ARITHM::minkowski_sum2(a,b,m) << " " << naive << "\\n";
  }
}
''')
        cls.executable = directory / 'probe'
        templates = Path(__file__).resolve().parents[3] / 'madgraph/iolibs/template_files/madmatrix'
        subprocess.run(['g++', '-std=c++20', '-O3', '-ffast-math',
                        '-I' + str(templates), str(source), '-o', str(cls.executable)],
                       check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def evaluate(self, cases):
        inputs = '\n'.join(' '.join(map(repr, a + b + [mass]))
                           for a, b, mass in cases) + '\n'
        result = subprocess.run([str(self.executable)], input=inputs,
                                capture_output=True, text=True, check=True)
        values = [tuple(map(float, line.split())) for line in result.stdout.splitlines()]
        self.assertEqual(len(values), len(cases))
        return values

    def reference(self, a, b, mass):
        with localcontext() as context:
            context.prec = 100
            q = [Decimal.from_float(x) + Decimal.from_float(y)
                 for x, y in zip(a, b)]
            return float(q[0]**2 - sum(x**2 for x in q[1:]) - Decimal.from_float(mass)**2)

    def test_near_collinear_gluon_propagator(self):
        # The actual parent momenta of the failed gg>ttbar+3g point 2791.
        a = [-500.0, 0.0, 0.0, 500.0]
        b = [50.13558898124127, 0.07492950381686886,
             -0.338395341987872, -50.13439095728541]
        compensated, naive = self.evaluate([(a, b, 0.0)])[0]
        reference = self.reference(a, b, 0.0)
        self.assertLessEqual(abs(compensated - reference), 4 * math.ulp(reference))
        self.assertGreater(abs(naive - reference), 100 * math.ulp(reference))

    def test_signed_and_massive_invariants_against_exact_inputs(self):
        random_source = random.Random(1103)
        cases = [([256.0, 0.0, 0.0, 256.0],
                  [256.0, 0.0, 0.0, -256.0], 512.0)]
        for _ in range(1000):
            energy = random_source.uniform(1, 400)
            angle = 10**random_source.uniform(-6, -2)
            px = energy * angle
            py = px * random_source.uniform(-1, 1)
            pz = -math.sqrt(energy**2 - px**2 - py**2)
            cases.append(([-500.0, 0.0, 0.0, 500.0], [energy, px, py, pz], 0.0))
            cases.append(([energy, px, py, pz],
                          [energy, -px, -py, -pz], 2 * energy))
        for (a, b, mass), (observed, _) in zip(cases, self.evaluate(cases)):
            reference = self.reference(a, b, mass)
            self.assertLessEqual(abs(observed - reference), 8 * math.ulp(reference),
                                 (a, b, mass, observed, reference))


@unittest.skipUnless(shutil.which('g++'), 'requires a C++ compiler')
class TestInvariantInterfaces(unittest.TestCase):
    """Compile the actual writer expression with the production type headers."""
    @staticmethod
    def header_flags(templates, backend):
        variant = 'gpu' if backend == 'cuda' else ('cpu' if backend == 'scalar' else 'simd')
        return ['-I'+str(templates/'backend'/variant),
                '-I'+str(templates/'backend/common'), '-I'+str(templates)]

    def test_runtime_mixed_lanes_signed_parents_and_width(self):
        from madmatrix.model_handling import MadMatrixALOHAWriter
        writer = object.__new__(MadMatrixALOHAWriter)
        writer.outgoing = 1
        writer.particles = ['V', 'V', 'V']
        writer.get_momentum_conservation_sign = lambda: ['+', '-', '+']
        expression = writer.stable_vector_denominator()
        templates = Path(__file__).resolve().parents[3] / 'madgraph/iolibs/template_files/madmatrix'
        # Alternate ill-conditioned and ordinary lanes in the same SIMD call.
        cases = [([500., 0., 0., -500.],
                  [50.13558898124127, .07492950381686886,
                   -.338395341987872, -50.13439095728541], 0.),
                 ([-32.000000001, 0., 0., -32.], [32., 0., 0., -32.], 64.),
                 ([-128., 16., 8., 32.], [64., 4., 2., -16.], 0.),
                 ([-128., 16., 8., 32.], [64., 4., 2., -16.], 64.)]
        source_text = '''#include "MadMatrixConfig.h"
#ifdef MADMATRIX_HAS_VECTORS_H
#include "MadMatrixVectors.h"
#else
#include "MadMatrixCxtypes.h"
#endif
#include "MadMatrixDenom.h"
#include <iostream>
#include <iomanip>
using namespace madmatrix;
int main() {
  double a[4][4], b[4][4], mass;
  while(std::cin >> mass) {
    for(int j=0;j<4;++j) for(int k=0;k<4;++k) std::cin >> a[j][k];
    for(int j=0;j<4;++j) for(int k=0;k<4;++k) std::cin >> b[j][k];
    struct Vector { fptype_momenta_sv pvec[4]; } V2, V3;
    fptype_denom_sv dP1[4];
    for(int k=0;k<4;++k) {
#ifdef MADMATRIX_SIMD
      for(int lane=0;lane<neppV;++lane) {
        V2.pvec[k][lane]=a[lane%4][k]; V3.pvec[k][lane]=b[lane%4][k];
      }
#else
      V2.pvec[k]=a[0][k]; V3.pvec[k]=b[0][k];
#endif
      dP1[k]=-V2.pvec[k]+V3.pvec[k];
    }
    const fptype M1=mass, W1=1.25;
    const cxtype_denom cId(0,1);
    const cxtype_amp_sv value = '''+expression+''';
    const auto real=cxreal(value), imag=cximag(value);
    const auto ordinary=dP1[0]*dP1[0]-dP1[1]*dP1[1]-dP1[2]*dP1[2]-dP1[3]*dP1[3]-M1*M1;
    for(int lane=0;lane<neppV;++lane) {
      std::cout << std::setprecision(17)
#ifdef MADMATRIX_SIMD
        << real[lane] << " " << imag[lane] << " " << ordinary[lane]
#else
        << real << " " << imag << " " << ordinary
#endif
        << "\\n";
    }
  }
}
'''
        backends = [('scalar', ['-DMADMATRIX_NOARMNEON'])]
        features = Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
        if platform.machine().lower() in ('x86_64', 'amd64'):
            backends = [('scalar', ['-march=x86-64'])]
            if 'avx2' in features:
                backends.append(('avx2', ['-march=haswell']))
            if 'avx512f' in features:
                backends.append(('avx512', ['-march=skylake-avx512', '-mprefer-vector-width=512', '-DMADMATRIX_PVW512']))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            shutil.copytree(templates/'Arithmetics', directory/'Arithmetics')
            source = directory/'runtime.cc'; source.write_text(source_text)
            for backend, flags in backends:
                executable = directory/backend
                compiled = subprocess.run(['g++', '-std=c++20', '-O3', '-ffast-math', *flags,
                    '-DMADMATRIX_FPTYPE_DOUBLE', '-DMADMATRIX_FPTYPE2_DOUBLE',
                    *self.header_flags(templates, backend), str(source), '-o', str(executable)],
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                for rotation in range(4):
                    selected = cases[rotation:]+cases[:rotation]
                    for mass in (0., 64.):
                        inputs = ' '.join(map(repr, [mass]+
                            [x for a, _, _ in selected for x in a]+
                            [x for _, b, _ in selected for x in b]))+'\n'
                        result = subprocess.run([str(executable)], input=inputs,
                            capture_output=True, text=True, check=True)
                        lines = result.stdout.splitlines()
                        self.assertEqual(len(lines), {'scalar': 1, 'avx2': 4, 'avx512': 8}[backend])
                        for lane, line in enumerate(lines):
                            real, imag, ordinary = map(float, line.split())
                            a, b, _ = selected[lane%4]
                            signed = [-x for x in a]
                            reference = TestCompensatedInvariant.reference(self, signed, b, mass)
                            q = [x+y for x, y in zip(signed, b)]
                            scale = sum(x*x for x in q)+mass*mass
                            if abs(ordinary) < 2**-10*scale:
                                self.assertLessEqual(abs(real-reference), 8*math.ulp(reference),
                                    (backend, rotation, lane, mass, real, reference))
                            else:
                                self.assertEqual(real, ordinary)
                            self.assertEqual(imag, mass*1.25)

    def test_scalar_simd_and_cuda_precision_interfaces(self):
        from madmatrix.model_handling import MadMatrixALOHAWriter
        writer = object.__new__(MadMatrixALOHAWriter)
        writer.outgoing = 1
        writer.particles = ['V', 'V', 'V']
        writer.get_momentum_conservation_sign = lambda: ['+', '+', '+']
        expression = writer.stable_vector_denominator()
        templates = Path(__file__).resolve().parents[3] / 'madgraph/iolibs/template_files/madmatrix'
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory/'Arithmetics').mkdir()
            shutil.copy2(templates/'Arithmetics/Invariant.h', directory/'Arithmetics/Invariant.h')
            source = directory/'probe.cc'
            source.write_text('''#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#include "MadMatrixConfig.h"
#ifdef MADMATRIX_HAS_VECTORS_H
#include "MadMatrixVectors.h"
#else
#include "MadMatrixCxtypes.h"
#endif
#include "MadMatrixDenom.h"
#ifdef MADMATRIX_GPUIMPL
using namespace madmatrix;
__global__ void probe(cxtype_amp_sv* out, const fptype_momenta_sv* input) {
#else
using namespace madmatrix;
cxtype_amp_sv probe(const fptype_momenta_sv* input) {
#endif
  struct Vector { fptype_momenta_sv pvec[4]; } V1, V2, V3;
  for(int k=0;k<4;++k) { V2.pvec[k]=input[k]; V3.pvec[k]=input[k+4]; V1.pvec[k]=V2.pvec[k]+V3.pvec[k]; }
  const fptype_denom_sv dP1[4]={-V1.pvec[0],-V1.pvec[1],-V1.pvec[2],-V1.pvec[3]};
  const fptype M1=64, W1=1;
  const cxtype_denom cId(0,1);
#if defined(MADMATRIX_FPTYPE_DOUBLE) && !defined(MADMATRIX_SIMD_DENOM64)
  const cxtype_amp_sv value = '''+expression+''';
#else
  const cxtype_amp_sv value = virtuality(dP1, mass2(M1,W1));
#endif
#ifdef MADMATRIX_GPUIMPL
  out[0]=value;
#else
  return value;
#endif
}
''')
            precisions = [(['MADMATRIX_FPTYPE_DOUBLE','MADMATRIX_FPTYPE2_DOUBLE'], 'd'),
                          (['MADMATRIX_FPTYPE_DOUBLE','MADMATRIX_FPTYPE2_FLOAT'], 'm'),
                          (['MADMATRIX_FPTYPE_FLOAT','MADMATRIX_FPTYPE2_FLOAT'], 'f'),
                          (['MADMATRIX_FPTYPE_FLOAT','MADMATRIX_FPTYPE2_FLOAT',
                            'MADMATRIX_FPTYPE_MOMENTA_DOUBLE'], 'v')]
            backends = [('scalar', ['-DMADMATRIX_NOARMNEON'])]
            if platform.machine().lower() in ('x86_64','amd64'):
                backends = [('scalar', ['-march=x86-64']),
                            ('avx2', ['-march=haswell']),
                            ('avx512', ['-march=skylake-avx512','-mprefer-vector-width=512','-DMADMATRIX_PVW512'])]
            for backend, flags in backends:
                for defines, precision in precisions:
                    with self.subTest(backend=backend, precision=precision):
                        result = subprocess.run(['g++','-std=c++20','-O3','-ffast-math',
                            *flags,*['-D'+name for name in defines],*self.header_flags(templates, backend),
                            '-fsyntax-only',str(source)],capture_output=True,text=True,timeout=60)
                        self.assertEqual(result.returncode, 0, result.stderr)
            if shutil.which('nvcc'):
                cuda=directory/'probe.cu';cuda.write_text(source.read_text())
                result=subprocess.run(['nvcc','-std=c++20','--use_fast_math',
                    '-allow-unsupported-compiler','-arch=sm_90',
                    '-DMADMATRIX_FPTYPE_DOUBLE','-DMADMATRIX_FPTYPE2_DOUBLE',
                    *self.header_flags(templates, 'cuda'),'-c',str(cuda),'-o',str(directory/'probe.o')],
                    capture_output=True,text=True,timeout=90)
                self.assertEqual(result.returncode,0,result.stderr)
