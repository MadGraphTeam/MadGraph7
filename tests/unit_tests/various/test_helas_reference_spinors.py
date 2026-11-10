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
reference momentum, and of what output --axial_gauge=True uses them for."""
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
      double precision p(0:3), r(0:3), k(0:3), m, pp, rr, x(9), dmax(12)
      double precision zero(0:3), small, large
      double complex ps(4,4), es(4,4), s, c1, ci, tmp(4), cur(4)
      parameter (ci=(0d0,1d0))
      integer i, j, h, nsf, it, h2, hg, nzero, nseed
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
c       What --axial_gauge=True relies on. A massless vector of momentum k
c       in the gauge of the (massive) momentum p, and the fermion of
c       momentum p quantised along k: the vector decouples from one spin
c       state of the fermion, a different one for each of its helicities.
        k(1) = rr*sqrt(1-(2*x(7)-1)**2)*cos(6.2831853d0*x(8))
        k(2) = rr*sqrt(1-(2*x(7)-1)**2)*sin(6.2831853d0*x(8))
        k(3) = rr*(2*x(7)-1)
        k(0) = sqrt(k(1)**2+k(2)**2+k(3)**2)
        do nsf = -1, 1, 2
          do hg = -1, 1, 2
            call vxxxxxr(k, 0d0, hg, nsf, p, vc)
            call eslash(vc, es)
            nzero = 0
            small = 0d0
            large = 0d0
            do h = -1, 1, 2
              call oxxxxxr(p, m, h, 1, 7, k, ho)
              call ixxxxxr(p, m, h, -1, 7, k, hi)
              s = 0d0
              c1 = 0d0
              do i = 1, 4
                cur(i) = 0d0
                tmp(i) = 0d0
                do j = 1, 4
                  cur(i) = cur(i) + ho%W(j)*es(j,i)
                  tmp(i) = tmp(i) + es(i,j)*hi%W(j)
                enddo
                s = s + abs(cur(i))
                c1 = c1 + abs(tmp(i))
              enddo
              s = s/sqrt(p(0))
              c1 = c1/sqrt(p(0))
              if (dble(s).lt.1d-10.and.dble(c1).lt.1d-10) then
                nzero = nzero + 1
                small = max(small, dble(s), dble(c1))
              else
                large = max(dble(s), dble(c1))
              endif
            enddo
            dmax(10) = max(dmax(10), 1d0*abs(nzero-1))
            dmax(11) = max(dmax(11), small)
            if (large.lt.1d-3) dmax(12) = 1d0
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
      write(*,'(a,es12.4)') 'DECOUPLED_COUNT ', dmax(10)
      write(*,'(a,es12.4)') 'DECOUPLED ', dmax(11)
      write(*,'(a,es12.4)') 'COUPLED_VANISHES ', dmax(12)
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

c     the same for a (complex) polarisation vector
      subroutine eslash(vc, es)
      use aloha_object
      implicit none
      type(aloha) vc
      double complex es(4,4), ci
      parameter (ci=(0d0,1d0))
      es(:,:) = 0d0
      es(1,3) = vc%W(1)-vc%W(4)
      es(1,4) = -(vc%W(2)-ci*vc%W(3))
      es(2,3) = -(vc%W(2)+ci*vc%W(3))
      es(2,4) = vc%W(1)+vc%W(4)
      es(3,1) = vc%W(1)+vc%W(4)
      es(3,2) = vc%W(2)-ci*vc%W(3)
      es(4,1) = vc%W(2)+ci*vc%W(3)
      es(4,2) = vc%W(1)-vc%W(4)
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

    def test_vector_decouples_from_one_spin_state(self):
        """A massless vector in the gauge of the momentum of a massive
        fermion (VXXXXXR takes a massive reference) does not couple to one
        of the two spin states of that fermion quantised along the momentum
        of the vector, for each helicity of the vector: this is what removes
        the diagrams where the reference gluon of g g > t t~ + gluons is
        attached to the external top quark line"""
        self.assertEqual(self.results['DECOUPLED_COUNT'], 0.)
        self.assertLess(self.results['DECOUPLED'], 1e-12)
        self.assertEqual(self.results['COUPLED_VANISHES'], 0.)


if __name__ == '__main__':
    unittest.main()
