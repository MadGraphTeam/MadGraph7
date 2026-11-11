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
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Tests of IXXXXXR / OXXXXXR (aloha/template_files/aloha_functions.f), the
spinors of a massive fermion whose spin is quantised along the axis fixed by a
reference momentum (output --axial_gauge=True)."""
from __future__ import absolute_import
import os
import shutil
import subprocess
import tempfile
import unittest

import madgraph.various.misc as misc
from madgraph import MG5DIR

pjoin = os.path.join

# Random momenta p (mass m) and lightlike references r, covering six decades
# in |p| and |r| and the special directions of r (along +-z, along +-p).
# Prints, as "NAME value", the largest violation of each property.
DRIVER = """
      program test_reference_spinors
      use aloha_object
      implicit none
      type(aloha) fi(-1:1,-1:1), fo(-1:1,-1:1), gi(-1:1,-1:1)
      type(aloha) go(-1:1,-1:1), hi, ho, vc
      double precision p(0:3), r(0:3), k(0:3), m, pp, rr, x(9), dmax(9)
      double precision zero(0:3)
      double complex ps(4,4), s, c1, ci, tmp(4)
      parameter (ci=(0d0,1d0))
      integer i, j, h, nsf, it, h2, nseed
      integer, allocatable :: seed(:)
c     the same momenta at every run
      call random_seed(size=nseed)
      allocate(seed(nseed))
      do i = 1, nseed
        seed(i) = 4242 + 37*i
      enddo
      call random_seed(put=seed)
      dmax(:) = 0d0
      zero(:) = 0d0
      do it = 1, 4000
        call random_number(x)
        m = 173d0
        if (mod(it,3).eq.0) m = 4.7d0
        pp = 10d0**(6d0*x(1)-2d0)
        p(1) = pp*sqrt(1-(2*x(2)-1)**2)*cos(6.2831853d0*x(3))
        p(2) = pp*sqrt(1-(2*x(2)-1)**2)*sin(6.2831853d0*x(3))
        p(3) = pp*(2*x(2)-1)
        p(0) = sqrt(pp**2+m**2)
        rr = 10d0**(6d0*x(4)-2d0)
        r(1) = rr*sqrt(1-(2*x(5)-1)**2)*cos(6.2831853d0*x(6))
        r(2) = rr*sqrt(1-(2*x(5)-1)**2)*sin(6.2831853d0*x(6))
        r(3) = rr*(2*x(5)-1)
        if (it.le.40) then
          r(1) = 0d0
          r(2) = 0d0
          r(3) = rr*(-1)**it
          if (it.gt.20) r(1:3) = p(1:3)/pp*rr*(-1)**it
        endif
        r(0) = sqrt(r(1)**2+r(2)**2+r(3)**2)
        call slash(p, ps)
        do nsf = -1, 1, 2
          do h = -1, 1, 2
            call ixxxxxr(p, m, h, nsf, 7, r, fi(h,nsf))
            call oxxxxxr(p, m, h, nsf, 7, r, fo(h,nsf))
            call ixxxxx(p, m, h, nsf, 7, gi(h,nsf))
            call oxxxxx(p, m, h, nsf, 7, go(h,nsf))
c           the momentum and the flavor are those of ixxxxx / oxxxxx
            do i = 0, 3
              dmax(9) = max(dmax(9), abs(fi(h,nsf)%P(i)-gi(h,nsf)%P(i)),
     &                   abs(fo(h,nsf)%P(i)-go(h,nsf)%P(i)))
            enddo
            dmax(9) = max(dmax(9), 1d0*abs(fi(h,nsf)%flv_index-7),
     &                 1d0*abs(fo(h,nsf)%flv_index-7))
c           Dirac equation: (pslash - nsf m) fi = 0, fo (pslash - nsf m) = 0
            do i = 1, 4
              s = -nsf*m*fi(h,nsf)%W(i)
              c1 = -nsf*m*fo(h,nsf)%W(i)
              do j = 1, 4
                s = s + ps(i,j)*fi(h,nsf)%W(j)
                c1 = c1 + fo(h,nsf)%W(j)*ps(j,i)
              enddo
              dmax(1) = max(dmax(1), abs(s)/p(0)**1.5d0,
     &                   abs(c1)/p(0)**1.5d0)
            enddo
c           fo = fi^dagger gamma0
            do i = 1, 4
              j = mod(i+1,4)+1
              dmax(4) = max(dmax(4), abs(fo(h,nsf)%W(i)
     &             -dconjg(fi(h,nsf)%W(j)))/sqrt(p(0)))
            enddo
c           a massless fermion and a vanishing reference: ixxxxx / oxxxxx
            call ixxxxxr(r, 0d0, h, nsf, 7, p, hi)
            call ixxxxx(r, 0d0, h, nsf, 7, vc)
            call oxxxxxr(r, 0d0, h, nsf, 7, p, ho)
            do i = 1, 4
              dmax(8) = max(dmax(8), abs(hi%W(i)-vc%W(i)))
            enddo
            call oxxxxx(r, 0d0, h, nsf, 7, vc)
            do i = 1, 4
              dmax(8) = max(dmax(8), abs(ho%W(i)-vc%W(i)))
            enddo
            call ixxxxxr(p, m, h, nsf, 7, zero, hi)
            call oxxxxxr(p, m, h, nsf, 7, zero, ho)
            do i = 1, 4
              dmax(8) = max(dmax(8), abs(hi%W(i)-gi(h,nsf)%W(i)),
     &                   abs(ho%W(i)-go(h,nsf)%W(i)))
            enddo
          enddo
c         normalisation and orthogonality: fo(h) fi(h') = 2 m nsf delta
          do h = -1, 1, 2
            do h2 = -1, 1, 2
              s = 0d0
              do i = 1, 4
                s = s + fo(h,nsf)%W(i)*fi(h2,nsf)%W(i)
              enddo
              if (h.eq.h2) s = s - 2d0*m*nsf
              dmax(2) = max(dmax(2), abs(s)/(2*m))
            enddo
          enddo
c         completeness: sum_h fi fo = pslash + nsf m
          do i = 1, 4
            do j = 1, 4
              s = fi(1,nsf)%W(i)*fo(1,nsf)%W(j)
     &          + fi(-1,nsf)%W(i)*fo(-1,nsf)%W(j) - ps(i,j)
              if (i.eq.j) s = s - nsf*m
              dmax(3) = max(dmax(3), abs(s)/p(0))
            enddo
          enddo
        enddo
c       charge conjugation: v = C ubar^T with the matrix (and the sign) that
c       relates ixxxxx(nsf=-1) to oxxxxx(nsf=+1), as a flipped fermion flow
c       needs
        do h = -1, 1, 2
          tmp(1) = -go(h,1)%W(2)
          tmp(2) = go(h,1)%W(1)
          tmp(3) = go(h,1)%W(4)
          tmp(4) = -go(h,1)%W(3)
          do i = 1, 4
            dmax(5) = max(dmax(5), abs(tmp(i)+gi(h,-1)%W(i))/sqrt(p(0)))
          enddo
          tmp(1) = -fo(h,1)%W(2)
          tmp(2) = fo(h,1)%W(1)
          tmp(3) = fo(h,1)%W(4)
          tmp(4) = -fo(h,1)%W(3)
          do i = 1, 4
            dmax(6) = max(dmax(6), abs(tmp(i)+fi(h,-1)%W(i))/sqrt(p(0)))
          enddo
        enddo
c       r = (|p|,-p): the helicity states of ixxxxx, up to a phase
        k(0) = pp
        k(1:3) = -p(1:3)
        do h = -1, 1, 2
          call ixxxxxr(p, m, h, 1, 7, k, hi)
          do h2 = -1, 1, 2
            s = 0d0
            do i = 1, 4
              s = s + go(h2,1)%W(i)*hi%W(i)
            enddo
            if (h.eq.h2) then
              dmax(7) = max(dmax(7), abs(abs(s)-2d0*m)/(2*m))
            else
              dmax(7) = max(dmax(7), abs(s)/(2*m))
            endif
          enddo
        enddo
      enddo
      write(*,'(a,es12.4)') 'DIRAC ', dmax(1)
      write(*,'(a,es12.4)') 'NORM ', dmax(2)
      write(*,'(a,es12.4)') 'COMPLETENESS ', dmax(3)
      write(*,'(a,es12.4)') 'CONJUGATE ', dmax(4)
      write(*,'(a,es12.4)') 'CHARGECONJ_HELAS ', dmax(5)
      write(*,'(a,es12.4)') 'CHARGECONJ ', dmax(6)
      write(*,'(a,es12.4)') 'HELICITY ', dmax(7)
      write(*,'(a,es12.4)') 'FALLBACK ', dmax(8)
      write(*,'(a,es12.4)') 'MOMENTUM ', dmax(9)
      end

c     pslash in the representation of the HELAS spinors
      subroutine slash(p, ps)
      implicit none
      double precision p(0:3)
      double complex ps(4,4), ci
      parameter (ci=(0d0,1d0))
      ps(:,:) = 0d0
      ps(1,3) = p(0)-p(3)
      ps(1,4) = -(p(1)-ci*p(2))
      ps(2,3) = -(p(1)+ci*p(2))
      ps(2,4) = p(0)+p(3)
      ps(3,1) = p(0)+p(3)
      ps(3,2) = p(1)-ci*p(2)
      ps(4,1) = p(1)+ci*p(2)
      ps(4,2) = p(0)-p(3)
      end
"""


def helas_source():
    """The ALOHA module and the external wavefunctions of the HELAS library,
    as the processes get them."""
    text = open(pjoin(MG5DIR, 'aloha', 'template_files',
                      'aloha_functions.f')).read()
    start = text.index('      module ALOHA_OBJECT')
    return text[start:text.index('      subroutine onia_proj(')]


@unittest.skipIf(misc.which('gfortran') is None, 'gfortran not available')
class TestReferenceSpinors(unittest.TestCase):
    """IXXXXXR and OXXXXXR are the spinors of a massive fermion in another
    spin basis: they have every property of IXXXXX and OXXXXX that the sum
    over the helicities of a matrix element relies on."""

    results = None

    @classmethod
    def setUpClass(cls):
        path = tempfile.mkdtemp(prefix='test_reference_spinors_')
        try:
            with open(pjoin(path, 'helas.f'), 'w') as fsock:
                fsock.write(helas_source())
            with open(pjoin(path, 'driver.f'), 'w') as fsock:
                fsock.write(DRIVER)
            subprocess.check_output(
                ['gfortran', '-O', '-ffixed-line-length-132', '-o', 'driver',
                 'helas.f', 'driver.f'], cwd=path, stderr=subprocess.STDOUT)
            out = subprocess.check_output([pjoin(path, 'driver')], cwd=path)
        finally:
            shutil.rmtree(path)
        cls.results = dict((line.split()[0], float(line.split()[1]))
                           for line in out.decode().splitlines() if line.split())

    def test_dirac_equation(self):
        self.assertLess(self.results['DIRAC'], 1e-12)

    def test_normalisation_and_completeness(self):
        """ubar(h) u(h') = 2 m delta and sum_h u ubar = pslash + m: the sum
        over the helicities of |M|^2 is the one of the helicity basis"""
        self.assertLess(self.results['NORM'], 1e-12)
        self.assertLess(self.results['COMPLETENESS'], 1e-12)

    def test_conjugate_spinor(self):
        """OXXXXXR is the Dirac conjugate of IXXXXXR, and the two are related
        by charge conjugation exactly as IXXXXX and OXXXXX are (a diagram
        written with a flipped fermion flow uses one for the other)"""
        self.assertEqual(self.results['CONJUGATE'], 0.)
        self.assertLess(self.results['CHARGECONJ_HELAS'], 1e-12)
        self.assertLess(self.results['CHARGECONJ'], 1e-12)

    def test_helicity_basis_is_a_special_case(self):
        """r = (|p|, -p) gives back the helicity states, up to a phase"""
        self.assertLess(self.results['HELICITY'], 1e-12)

    def test_fallback(self):
        """a massless fermion or no reference: IXXXXX / OXXXXX, bit for bit;
        the momentum and the flavor of the wavefunction never change"""
        self.assertEqual(self.results['FALLBACK'], 0.)
        self.assertEqual(self.results['MOMENTUM'], 0.)



class TestPythonReferenceSpinors(unittest.TestCase):
    """The Python ixxxxxr / oxxxxxr / vxxxxxr (aloha/template_files/
    wavefunctions.py), used by check gauge and check brs: the same
    properties as the Fortran routines."""

    @staticmethod
    def momentum(rng, m, pp):
        import math
        ct = rng.uniform(-1, 1)
        ph = rng.uniform(0, 2 * math.pi)
        st = math.sqrt(1 - ct * ct)
        return [math.sqrt(pp * pp + m * m), pp * st * math.cos(ph),
                pp * st * math.sin(ph), pp * ct]

    @staticmethod
    def slash(p):
        ps = [[0j] * 4 for _ in range(4)]
        ps[0][2], ps[0][3] = p[0] - p[3], -(p[1] - 1j * p[2])
        ps[1][2], ps[1][3] = -(p[1] + 1j * p[2]), p[0] + p[3]
        ps[2][0], ps[2][1] = p[0] + p[3], p[1] - 1j * p[2]
        ps[3][0], ps[3][1] = p[1] + 1j * p[2], p[0] - p[3]
        return ps

    def test_spinors(self):
        """Dirac equation, normalisation, completeness, Dirac conjugate,
        helicity basis for r = (|p|,-p), and ixxxxx / oxxxxx without a
        reference or for a massless fermion"""

        import math
        import random
        from aloha.template_files.wavefunctions import ixxxxx, oxxxxx, \
                                                       ixxxxxr, oxxxxxr
        rng = random.Random(4242)
        worst = dict.fromkeys(('dirac', 'norm', 'complete', 'conj', 'hel'), 0.)
        for _ in range(300):
            m = 173. if rng.random() < 0.7 else 4.7
            p = self.momentum(rng, m, 10 ** rng.uniform(-2, 4))
            r = self.momentum(rng, 0., 10 ** rng.uniform(-2, 4))
            ps = self.slash(p)
            for nsf in (-1, 1):
                fi = dict((h, ixxxxxr(p, m, h, nsf, r)) for h in (-1, 1))
                fo = dict((h, oxxxxxr(p, m, h, nsf, r)) for h in (-1, 1))
                for h in (-1, 1):
                    w = [fi[h][2 + i] for i in range(4)]
                    for i in range(4):
                        res = sum(ps[i][j] * w[j] for j in range(4)) - nsf * m * w[i]
                        worst['dirac'] = max(worst['dirac'], abs(res) / p[0] ** 1.5)
                        j = (i + 2) % 4
                        worst['conj'] = max(worst['conj'], abs(fo[h][2 + i] -
                                            complex(fi[h][2 + j]).conjugate()))
                    for h2 in (-1, 1):
                        s = sum(fo[h][2 + i] * fi[h2][2 + i] for i in range(4))
                        s -= 2 * m * nsf if h == h2 else 0
                        worst['norm'] = max(worst['norm'], abs(s) / (2 * m))
                for i in range(4):
                    for j in range(4):
                        s = sum(fi[h][2 + i] * fo[h][2 + j] for h in (-1, 1))
                        s -= ps[i][j] + (nsf * m if i == j else 0)
                        worst['complete'] = max(worst['complete'], abs(s) / p[0])
            k = [math.sqrt(sum(x * x for x in p[1:]))] + [-x for x in p[1:]]
            for h in (-1, 1):
                a, b = ixxxxxr(p, m, h, 1, k), oxxxxx(p, m, h, 1)
                s = sum(b[2 + i] * a[2 + i] for i in range(4))
                worst['hel'] = max(worst['hel'], abs(abs(s) - 2 * m) / (2 * m))
            for h in (-1, 1):
                self.assertEqual(list(ixxxxxr(p, m, h, 1, [0., 0., 0., 0.])),
                                 list(ixxxxx(p, m, h, 1)))
                self.assertEqual(list(oxxxxxr(r, 0., h, 1, p)),
                                 list(oxxxxx(r, 0., h, 1)))
        for key, value in worst.items():
            self.assertLess(value, 1e-12, key)

    def test_vector(self):
        """vxxxxxr: r.epsilon = 0 and p.epsilon = 0 for a massless vector,
        vxxxxx for a massive one and for the BRS polarisation (nhel = 4)"""

        import random
        from aloha.template_files.wavefunctions import vxxxxx, vxxxxxr
        rng = random.Random(7)
        dot = lambda a, b: a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
        for _ in range(100):
            k = self.momentum(rng, 0., rng.uniform(1, 1000))
            r = self.momentum(rng, 0., rng.uniform(1, 1000))
            for h in (-1, 1):
                for nsv in (-1, 1):
                    vc = vxxxxxr(k, 0., h, nsv, r)
                    eps = [vc[2 + i] for i in range(4)]
                    self.assertLess(abs(dot(r, eps)) / r[0], 1e-12)
                    self.assertLess(abs(dot(k, eps)) / k[0], 1e-12)
            w = self.momentum(rng, 80.4, rng.uniform(1, 1000))
            self.assertEqual(list(vxxxxxr(w, 80.4, 1, 1, r)), list(vxxxxx(w, 80.4, 1, 1)))
            self.assertEqual(list(vxxxxxr(k, 0., 4, 1, r)), list(vxxxxx(k, 0., 4, 1)))

if __name__ == '__main__':
    unittest.main()
