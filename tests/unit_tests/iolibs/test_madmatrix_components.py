"""Compiled squared-order contractions retain the matrix-element precision."""
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('g++'), 'requires a C++ compiler')
class TestSquaredOrderPrecision(unittest.TestCase):
    def test_positive_color_matrix_cancellation(self):
        templates = Path(__file__).resolve().parents[3] / 'madgraph/iolibs/template_files/madmatrix'
        variants = [('scalar', 'cpu', [])]
        features = Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
        if platform.machine().lower() in ('x86_64', 'amd64'):
            variants = [('scalar', 'cpu', ['-march=x86-64'])]
            if 'avx2' in features:
                variants.append(('avx2', 'simd', ['-march=haswell']))
            if 'avx512f' in features:
                variants.append(('avx512', 'simd', ['-march=skylake-avx512', '-DMADMATRIX_PVW512']))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'ProcessData.h').write_text('''#pragma once
namespace madmatrix { namespace ProcessData {
constexpr int ncolor=3, nampso=2, nsqampso=3, njampso=6;
constexpr int np4=4, nw6=6, npar=4, nparf=2, ndiagrams=1;
} }
''')
            (directory/'Parameters.h').write_text('''#pragma once
namespace madmatrix { struct Parameters_dependentCouplings { static constexpr int ndcoup=1; }; }
''')
            # This is a positive semidefinite Gram matrix: ||(x-z,y)||^2.
            # With jamp=(10000,1,10000), the exact contraction is one. Its
            # colour terms (-1e8,1,1e8) must be promoted before summation.
            (directory/'ColorData.h').write_text('''#pragma once
#include "ProcessData.h"
namespace madmatrix { namespace ColorMatrixData {
using namespace ProcessData;
constexpr fptype_colour colorMatrix[3][3]={{1,0,-1},{0,1,0},{-1,0,1}};
constexpr fptype_colour colorDenom[3]={1,1,1};
constexpr int sqSoIndex[2][2]={{0,1},{1,2}};
constexpr bool chosenSqso[3]={true,false,false};
} }
''')
            source = directory/'probe.cc'
            source.write_text('''#include "color_sum.h"
#include <iostream>
using namespace madmatrix;
int main() {
#ifdef MADMATRIX_SIMD
  constexpr int lanes=neppV;
#else
  constexpr int lanes=1;
#endif
#if defined MADMATRIX_SIMD && defined MADMATRIX_FPTYPE2_FLOAT
  constexpr int halves=2;
#else
  constexpr int halves=1;
#endif
  cxtype_amp_sv jamps[halves*ProcessData::njampso] = {};
  for(int half=0;half<halves;++half) for(int color=0;color<3;++color) {
    fptype_amp_sv real={}, imag={};
    const fptype value=color==1 ? 1 : 10000;
#ifdef MADMATRIX_SIMD
    for(int lane=0;lane<lanes;++lane) real[lane]=value;
#else
    real=value;
#endif
    jamps[half*ProcessData::njampso+color]=cxmake(real,imag);
  }
  alignas(64) fptype total[16]={}, components[3*16]={};
  color_sum_cpu(total,jamps,0,components,16);
  for(int event=0;event<halves*lanes;++event) {
    std::cout << total[event] << " " << components[event] << "\\n";
    if(total[event]!=1 || components[event]!=1 || components[16+event]!=0 || components[32+event]!=0) return 1;
  }
}
''')
            for name, backend, flags in variants:
                for color_precision in ['DOUBLE', 'FLOAT']:
                    with self.subTest(backend=name, color_precision=color_precision):
                        executable = directory/(name+'-'+color_precision)
                        command = ['g++', '-std=c++20', '-O3', '-ffast-math', *flags,
                            '-DMADMATRIX_FPTYPE_DOUBLE', '-DMADMATRIX_FPTYPE2_'+color_precision,
                            '-I'+str(directory), '-I'+str(templates/'backend'/backend),
                            '-I'+str(templates/'backend/common'), str(source),
                            str(templates/'backend'/backend/'color_sum.cc'), '-o', str(executable)]
                        compiled = subprocess.run(command, capture_output=True, text=True, timeout=60)
                        self.assertEqual(compiled.returncode, 0, compiled.stderr)
                        result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
                        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                        expected_lanes = {'scalar': 1, 'avx2': 4, 'avx512': 8}[name]
                        expected_count = expected_lanes*(2 if backend=='simd' and color_precision=='FLOAT' else 1)
                        self.assertEqual(len(result.stdout.splitlines()), expected_count)
