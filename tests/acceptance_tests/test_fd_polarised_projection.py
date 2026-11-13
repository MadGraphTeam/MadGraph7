################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
################################################################################
"""The FD-gauge polarised-propagator projection in the three library languages.

fd_polarized_propagator projects the off-shell FD wavefunction of a massive
vector on one piece of the five-component FD numerator ({0}, {T}, {+}, {-},
{A}, {S}, {0S}; aloha_writers.FD_POLARIZED_PROPAGATORS). It exists three times,
in aloha_functions_fd.f (Fortran outputs), aloha_aux_functions.fd_cc (the C++
standalone of 'check language') and wavefunctions.py (the Python evaluator);
madmatrix inlines it and is compared with the Fortran output elsewhere
(test_fd_polarised_propagator_madmatrix_vs_fortran).

Here the Fortran and C++ libraries are compiled on their own with a small
driver and compared, piece by piece, with the Python routine on the same
wavefunctions: generic momenta with either sign of the energy, a line exactly
at rest with either sign, and one at rest up to rounding noise (where the
helicity axis is pinned to +z).

Run locally with::

    ./tests/test_manager.py test_fd_polarised_projection_languages -pA -t0
"""

from __future__ import absolute_import
from __future__ import division

import os
import random
import shutil
import subprocess
import tempfile
import unittest

import madgraph
import aloha.template_files.wavefunctions as wavefunctions

pjoin = os.path.join
TEMPLATES = pjoin(madgraph.MG5DIR, 'aloha', 'template_files')

FORTRAN_DRIVER = """
      program driver
      use aloha_object
      implicit none
      type(aloha) w
      double precision m, width, re(5), im(5)
      integer ipol, i, ncase, k
      read(*,*) ncase
      do k = 1, ncase
         read(*,*) w%P(0), w%P(1), w%P(2), w%P(3), m, width, ipol
         read(*,*) (re(i), im(i), i = 1, 5)
         do i = 1, 5
            w%W(i) = dcmplx(re(i), im(i))
         enddo
         call fd_polarized_propagator(w, m, width, ipol)
         write(*,'(10ES26.17)') (dble(w%W(i)), dimag(w%W(i)), i = 1, 5)
      enddo
      end
"""

CPP_DRIVER = """
#include <cstdio>
#include <complex>
#include "aloha_aux_functions.h"
int main() {
  int ncase;
  if (std::scanf("%d", &ncase) != 1) return 1;
  for (int k = 0; k < ncase; ++k) {
    ALOHAOBJ w;
    double m, width, re, im;
    int ipol;
    std::scanf("%lf %lf %lf %lf %lf %lf %d", &w.p[0], &w.p[1], &w.p[2], &w.p[3],
               &m, &width, &ipol);
    for (int i = 0; i < 5; ++i) {
      std::scanf("%lf %lf", &re, &im);
      w.W[i] = std::complex<double>(re, im);
    }
    fd_polarized_propagator(w, m, width, ipol);
    for (int i = 0; i < 5; ++i)
      std::printf("%.17e %.17e ", std::real(w.W[i]), std::imag(w.W[i]));
    std::printf("\\n");
  }
  return 0;
}
"""


def cases():
    """(momentum, mass, width, ipol, five wavefunction components)"""
    rng = random.Random(11)
    momenta = []
    for sign in (1, -1):
        for _ in range(2):
            vec = [rng.uniform(-200, 200) for _ in range(3)]
            energy = (sum(x * x for x in vec) + rng.uniform(30, 150)**2)**0.5
            momenta.append([sign * energy] + vec)
        momenta.append([sign * 80.0, 0., 0., 0.])                  # at rest
        momenta.append([sign * 81.0, 3e-12, -2e-12, 1e-13])        # rounding
    out = []
    for mom in momenta:
        for ipol in range(7):
            wf = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(5)]
            out.append((mom, 80.4, 2.05, ipol, wf))
    return out


def python_values(case):
    mom, mass, width, ipol, wf = case
    win = wavefunctions.WaveFunction(size=7)
    win.momenta = list(mom)
    for i in range(5):
        win[i + 2] = wf[i]
    wout = wavefunctions.fd_polarized_propagator(win, mass, width, ipol)
    return [wout[i + 2] for i in range(5)]


def driver_input(all_cases):
    lines = ['%d' % len(all_cases)]
    for mom, mass, width, ipol, wf in all_cases:
        lines.append(' '.join('%.17e' % x for x in mom + [mass, width]) + ' %d' % ipol)
        lines.append(' '.join('%.17e %.17e' % (z.real, z.imag) for z in wf))
    return '\n'.join(lines) + '\n'


def parse(output):
    values = []
    for line in output.strip().splitlines():
        numbers = [float(x.replace('D', 'E')) for x in line.split()]
        values.append([complex(numbers[2 * i], numbers[2 * i + 1]) for i in range(5)])
    return values


class TestFDPolarisedProjectionLanguages(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='fd_pol_projection_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def run_driver(self, command, all_cases):
        result = subprocess.run(command, input=driver_input(all_cases),
                                capture_output=True, text=True, cwd=self.tmpdir)
        self.assertEqual(result.returncode, 0, result.stderr)
        return parse(result.stdout)

    def build_fortran(self):
        if not shutil.which('gfortran'):
            self.skipTest('no gfortran')
        shutil.copy(pjoin(TEMPLATES, 'aloha_functions_fd.f'), self.tmpdir)
        with open(pjoin(self.tmpdir, 'driver.f'), 'w') as fsock:
            fsock.write(FORTRAN_DRIVER)
        for command in (['gfortran', '-c', '-w', '-ffixed-line-length-132',
                         'aloha_functions_fd.f'],
                        ['gfortran', '-w', '-ffixed-line-length-132', '-o',
                         'fdriver', 'driver.f', 'aloha_functions_fd.o']):
            result = subprocess.run(command, capture_output=True, text=True,
                                    cwd=self.tmpdir)
            self.assertEqual(result.returncode, 0, result.stderr)
        return ['./fdriver']

    def build_cpp(self):
        compiler = shutil.which('c++') or shutil.which('g++')
        if not compiler:
            self.skipTest('no C++ compiler')
        shutil.copy(pjoin(TEMPLATES, 'aloha_aux_functions.fd_h'),
                    pjoin(self.tmpdir, 'aloha_aux_functions.h'))
        shutil.copy(pjoin(TEMPLATES, 'aloha_aux_functions.fd_cc'),
                    pjoin(self.tmpdir, 'aloha_aux_functions.cc'))
        with open(pjoin(self.tmpdir, 'driver.cc'), 'w') as fsock:
            fsock.write(CPP_DRIVER)
        result = subprocess.run([compiler, '-std=c++17', '-O1', '-o', 'cdriver',
                                 'driver.cc', 'aloha_aux_functions.cc'],
                                capture_output=True, text=True, cwd=self.tmpdir)
        self.assertEqual(result.returncode, 0, result.stderr)
        return ['./cdriver']

    def compare(self, language, values, all_cases):
        self.assertEqual(len(values), len(all_cases), language)
        for case, value in zip(all_cases, values):
            reference = python_values(case)
            scale = max(1e-300, max(abs(z) for z in case[4]))
            for i in range(5):
                self.assertLess(abs(value[i] - reference[i]) / scale, 1e-12,
                                '%s, momentum %s, ipol %d, component %d: %r vs %r'
                                % (language, case[0], case[3], i, value[i],
                                   reference[i]))

    def test_fd_polarised_projection_languages(self):
        """Fortran and C++ fd_polarized_propagator agree with the Python one,
        for every piece and both signs of the energy, at rest included."""
        all_cases = cases()
        self.compare('Fortran', self.run_driver(self.build_fortran(), all_cases),
                     all_cases)
        self.compare('C++', self.run_driver(self.build_cpp(), all_cases),
                     all_cases)

    def test_fd_polarised_projection_at_rest(self):
        """A line at rest (exactly, or up to rounding), with either sign of
        the energy flowing through it: the helicity axis is +z in every case.
        A circular vector around z then lies entirely in one helicity piece,
        the same one in the four cases (before, the axis followed the
        rounding noise, and its sign the sign of the energy)."""
        mass, width = 80.4, 2.05
        circular = [0., -2**-0.5, -1j * 2**-0.5, 0., 0.]
        pieces = []
        for mom in ([80.0, 0., 0., 0.], [-80.0, 0., 0., 0.],
                    [81.0, 3e-12, -2e-12, 1e-13], [-81.0, -3e-12, 2e-12, -1e-13]):
            win = wavefunctions.WaveFunction(size=7)
            win.momenta = mom
            for i in range(5):
                win[i + 2] = circular[i]
            got = dict((ipol, [wavefunctions.fd_polarized_propagator(
                                   win, mass, width, ipol)[i + 2] for i in range(5)])
                       for ipol in (1, 2, 3))
            # the vector is transverse to z ...
            for i in range(5):
                self.assertAlmostEqual(abs(got[1][i] - circular[i]), 0., 12, str(mom))
            # ... and of one helicity: one piece keeps it, the other is zero
            norms = sorted(sum(abs(x) for x in got[ipol]) for ipol in (2, 3))
            self.assertAlmostEqual(norms[0], 0., 12, str(mom))
            pieces.append(got)
        for got in pieces[1:]:
            for ipol in (2, 3):
                for i in range(5):
                    self.assertAlmostEqual(abs(got[ipol][i] - pieces[0][ipol][i]),
                                           0., 12)

if __name__ == '__main__':
    unittest.main()
