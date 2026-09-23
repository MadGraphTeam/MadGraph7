#!/usr/bin/env python3
"""Emit Source/PDF/lep_densities/<set>/bstrahl.f from the CIRCE parameters
of vendor/eepdfgrid/calcpdf_<collider>.f.

The dressed-lepton grids in eepdf.f are the Mellin convolution of a CIRCE
beamstrahlung spectrum with the ISR structure function.  bstrahl.f exposes
that convolution's integrand so the x = xbs * xisr split can be sampled at
event-writing time.
"""
import os
import sys

# The exponential factor of the CIRCE spectrum is collider dependent:
# vendor/eepdfgrid/calcpdf_<collider>.f writes it as
#   'sqrt' : dexp(q*dsqrt(u))         (cepc240, fccee240, fccee365, ilc250)
#   'u15'  : dexp(q*u**1.5d0)         (ilc500)
#   'ym15' : dexp(q*(x/z)**(-1.5d0))  (clic3000) -- note: in x/z, not in u
# with u = 1-x/z.  Getting this wrong is silent: the normalisation of the
# split drifts but nothing crashes, so test_bstrahl.f checks every set
# against Iapq() of the corresponding vendor file.
QTERM = {
    'sqrt': 'dexp(q*dsqrt(u))',
    'u15':  'dexp(q*u**1.5d0)',
    'ym15': 'dexp(q*y**(-1.5d0))',
}

# <density set> : (collider source, q-term form, a11, a01, a10, a00,
#                  p1, q1, p2, q2, p3, q3, p4, q4)
SETS = {
    'cepc240ll':  ('cepc240',  'sqrt', 0.8351,  0.3513,  0.3515,  0.1663,
                   -910.9, -17.28, -914.2, -17.27, -917.9, -15.44, -918.8, -15.45),
    'clic3000ll': ('clic3000', 'ym15', 0.7434,  0.04563, 0.04568, 0.003537,
                   -0.6016, -7.179e-3, -0.6089, -7.135e-3,
                   -0.5629, -7.464e-3, -0.5743, -7.259e-3),
    'fcce240ll':  ('fccee240', 'sqrt', 0.8698,  0.2863,  0.2853,  0.1092,
                   -901.2, -19.30, -916.5, -18.70, -899.2, -18.03, -918.7, -17.48),
    'fcce365ll':  ('fccee365', 'sqrt', 0.7883,  0.4056,  0.4059,  0.2281,
                   -714.3, -14.64, -714.5, -14.69, -714.7, -12.77, -717.1, -12.80),
    'ilc500ll':   ('ilc500',   'u15',  0.5012,  0.1613,  0.1613,  0.06379,
                   -8.514, -5.808, -8.505, -5.823, -7.535, -6.790, -7.481, -6.849),
}

# Sets with no beamstrahlung at all: pure ISR densities.
STUB_SETS = ['isronlyll', 'isrbetll', 'isrbetmll']


def d(x):
    """Fortran double literal."""
    s = repr(float(x))
    if 'e' in s:
        return s.replace('e', 'd')
    return s + 'd0'


HEADER = """c     ------------------------------------------------------------------
c     Beamstrahlung / ISR splitting for the '%(name)s' lepton density.
c
c     eepdf.f ships phibar(x,Q2), the Mellin convolution of a CIRCE
c     beamstrahlung spectrum with the ISR structure function.  Writing
c     z for the ISR momentum fraction and t = (z-x)/(1-x), the reduced
c     density of a beamstrahlung component is
c
c        phibar(x,Q2) = int_0^1 dt  a * exp(p*u) * %(qterm)s
c                                 * z**(-1/3) * isr_tilde(z,beta)
c                                 * t**(-2/3) * (1-t)**(beta-1)
c
c     with z = t*(1-x)+x and u = 1-x/z.  So x factorises as
c
c        x = xbs * xisr,     xisr = z,   xbs = x/z
c
c     where xisr is ordinary collinear ISR off the beam lepton and xbs
c     is the beam-energy loss caused by the field of the opposing
c     bunch.  Only xisr may be handed to a parton shower's backward
c     evolution; xbs is a property of the beam.  See sample_bstrahl.f.
c
c     The t integral is split at t=ra exactly as vendor/eepdfgrid does,
c     because the substitution that removes the endpoint singularities
c     leaves a smooth integrand in the sampling variable r, which is
c     what makes a uniform-grid CDF inversion accurate here.
c
c     Generated from vendor/eepdfgrid/calcpdf_%(collider)s.f by
c     gen_bstrahl.py -- do not edit by hand.
c     ------------------------------------------------------------------

"""

STUB_BODY = """c     ------------------------------------------------------------------
c     Beamstrahlung / ISR splitting for the '%(name)s' lepton density.
c
c     This density is pure ISR: there is no beamstrahlung to split off,
c     so the whole of x is ISR off a beam lepton at its nominal energy.
c     See sample_bstrahl.f for how this is consumed.
c     ------------------------------------------------------------------

      logical function eepdf_has_bstrahl()
      implicit none
      eepdf_has_bstrahl = .false.
      return
      end


      subroutine eepdf_sample_bstrahl(x,Q2,n,partonid,beamid,u1,u2,
     &                                xbs,xisr)
      implicit none
      double precision x, Q2, u1, u2, xbs, xisr
      integer n, partonid, beamid
      xbs = 1d0
      xisr = x
      return
      end
"""

BODY = """      logical function eepdf_has_bstrahl()
      implicit none
      eepdf_has_bstrahl = .true.
      return
      end


      subroutine eepdf_bstrahl_pars(n,partonid,beamid,a,p,q,has)
c     CIRCE parameters of the beamstrahlung factor of component n.
c     has=.false. for the components which carry no beamstrahlung on
c     this beam (the delta(1-xbs) pieces), mirroring eepdf_tilde_type.
      implicit none
      integer n, partonid, beamid
      double precision a, p, q
      logical has

      double precision a11,a01,a10,a00
      double precision p1,q1,p2,q2,p3,q3,p4,q4
      data a11/%(a11)s/,a01/%(a01)s/,a10/%(a10)s/,a00/%(a00)s/
      data p1/%(p1)s/,q1/%(q1)s/,p2/%(p2)s/,q2/%(q2)s/
      data p3/%(p3)s/,q3/%(q3)s/,p4/%(p4)s/,q4/%(q4)s/

      has = .false.
      a = 0d0
      p = 0d0
      q = 0d0

      if (partonid.ne.beamid) return

      if (beamid.eq.11) then
        if (n.eq.2) then
          a = a01/dsqrt(a11)
          p = p1
          q = q1
          has = .true.
        else if (n.eq.4) then
          a = dsqrt(a00)
          p = p3
          q = q3
          has = .true.
        endif
      else if (beamid.eq.-11) then
        if (n.eq.3) then
          a = a10/dsqrt(a11)
          p = p2
          q = q2
          has = .true.
        else if (n.eq.4) then
          a = dsqrt(a00)
          p = p4
          q = q4
          has = .true.
        endif
      endif

      return
      end


      subroutine eepdf_sample_bstrahl(x,Q2,n,partonid,beamid,u1,u2,
     &                                xbs,xisr)
c     Sample x = xbs*xisr for component n of the beam 'beamid', given
c     two uniform deviates u1 (picks the half of the t integral) and
c     u2 (inverts the CDF within that half).
      implicit none
      double precision x, Q2, u1, u2, xbs, xisr
      integer n, partonid, beamid

      integer ngrid
      parameter (ngrid=128)

      double precision me, alphaem, ra, xmax
      parameter (me=0.511d-3)
c     Gmu scheme, as in gridpdfaux.f / calcpdf_%(collider)s.f
      parameter (alphaem=0.007562397d0)
      parameter (ra=0.5d0)
c     above this x the split is numerically indistinguishable from 1
      parameter (xmax=1d0-1d-12)

      double precision a, p, q
      logical has
      double precision beta, pa, pb, rb, pi
      double precision cdf1(0:ngrid), cdf2(0:ngrid)
      double precision f0, f1, h, r, z, t, tot, target
      integer i, ilo, ihi, imid
      double precision bstrahl_integrand

      xbs = 1d0
      xisr = x

      call eepdf_bstrahl_pars(n,partonid,beamid,a,p,q,has)
      if (.not.has) return
      if (x.ge.xmax .or. x.le.0d0) return

      pi = 4d0*datan(1d0)
      beta = alphaem/pi * (dlog(Q2/me/me)-1d0)
      pa = -2d0/3d0
      pb = beta-1d0
      rb = 1d0-ra
      h = 1d0/dble(ngrid)

c     Cumulative (trapezoidal) of both halves over the sampling
c     variable r in [0,1].  The a prefactor is common to the two and
c     drops out of the normalisation, so it is left out here.
      cdf1(0) = 0d0
      cdf2(0) = 0d0
      f0 = bstrahl_integrand(0d0,1,x,p,q,beta,pa,pb,ra,rb)
      do i = 1, ngrid
        f1 = bstrahl_integrand(dble(i)*h,1,x,p,q,beta,pa,pb,ra,rb)
        cdf1(i) = cdf1(i-1) + 0.5d0*h*(f0+f1)
        f0 = f1
      enddo
      f0 = bstrahl_integrand(0d0,2,x,p,q,beta,pa,pb,ra,rb)
      do i = 1, ngrid
        f1 = bstrahl_integrand(dble(i)*h,2,x,p,q,beta,pa,pb,ra,rb)
        cdf2(i) = cdf2(i-1) + 0.5d0*h*(f0+f1)
        f0 = f1
      enddo

      tot = cdf1(ngrid)+cdf2(ngrid)
      if (tot.le.0d0) return

c     Pick the half, then invert its CDF by bisection plus a linear
c     interpolation inside the bin.
      if (u1*tot .lt. cdf1(ngrid)) then
        target = u2*cdf1(ngrid)
        ilo = 0
        ihi = ngrid
        do while (ihi-ilo .gt. 1)
          imid = (ilo+ihi)/2
          if (cdf1(imid).le.target) then
            ilo = imid
          else
            ihi = imid
          endif
        enddo
        r = dble(ilo)*h
        if (cdf1(ihi).gt.cdf1(ilo))
     &    r = r + h*(target-cdf1(ilo))/(cdf1(ihi)-cdf1(ilo))
        t = ra*r**(1d0/(pa+1d0))
      else
        target = u2*cdf2(ngrid)
        ilo = 0
        ihi = ngrid
        do while (ihi-ilo .gt. 1)
          imid = (ilo+ihi)/2
          if (cdf2(imid).le.target) then
            ilo = imid
          else
            ihi = imid
          endif
        enddo
        r = dble(ilo)*h
        if (cdf2(ihi).gt.cdf2(ilo))
     &    r = r + h*(target-cdf2(ilo))/(cdf2(ihi)-cdf2(ilo))
        t = 1d0-rb*r**(1d0/(pb+1d0))
      endif

      z = t*(1d0-x)+x
      if (z.le.x) then
        return
      else if (z.ge.1d0) then
        xisr = 1d0
        xbs = x
        return
      endif

      xisr = z
      xbs = x/z

      return
      end


      double precision function bstrahl_integrand(r,iterm,x,p,q,beta,
     &                                            pa,pb,ra,rb)
c     Integrand of the t integral in the sampling variable r, for the
c     half iterm (1: t in [0,ra], 2: t in [ra,1]).  Identical to
c     Iapqterm1/Iapqterm2 of vendor/eepdfgrid up to the overall a.
      implicit none
      double precision r, x, p, q, beta, pa, pb, ra, rb
      integer iterm
      double precision t, z, y, u, g
      double precision bstrahl_isr_tilde

      if (iterm.eq.1) then
        t = ra*r**(1d0/(pa+1d0))
      else
        t = 1d0-rb*r**(1d0/(pb+1d0))
      endif
      z = t*(1d0-x)+x
      if (z.le.0d0) then
        bstrahl_integrand = 0d0
        return
      endif
      y = x/z
      u = 1d0-y
      if (u.lt.0d0) u = 0d0

      g = dexp(p*u)*%(qterm)s*bstrahl_isr_tilde(z,beta)
      g = g*z**(-1d0/3d0)

      if (iterm.eq.1) then
        bstrahl_integrand = g*(1d0-t)**pb*ra**(pa+1d0)/(pa+1d0)
      else
        bstrahl_integrand = g*t**pa*rb**(pb+1d0)/(pb+1d0)
      endif

      return
      end


      double precision function bstrahl_isr_tilde(x,beta)
c     Copy of vendor/eepdfgrid/calcisr.f, kept local to this file so
c     that no density set depends on symbols from another.
      implicit none
      double precision x, beta
      double precision res, pi

      pi = 4d0*datan(1d0)
      res = 0d0
      if (x .lt. 0.9999999d0) then
        res = -beta*(1d0+x)/2d0
        res = res-beta*beta*(
     &     (1d0+3d0*x*x)/(1d0-x)*dlog(x)
     &     +4d0*(1d0+x)*dlog(1d0-x)+5d0+x)
     &     /8d0
        res = res*(1d0-x)**(1-beta)
      endif
      res = res+
     &  (1d0+3d0/4d0*beta+(27d0-8d0*pi*pi)/96d0*beta*beta)
     &  *beta
      bstrahl_isr_tilde = res

      return
      end
"""


def main(root):
    base = os.path.join(root, 'Template', 'Common', 'Source', 'PDF',
                        'lep_densities')
    for name, pars in SETS.items():
        collider, qform = pars[0], pars[1]
        subs = dict(name=name, collider=collider, qterm=QTERM[qform])
        keys = ['a11', 'a01', 'a10', 'a00',
                'p1', 'q1', 'p2', 'q2', 'p3', 'q3', 'p4', 'q4']
        for key, val in zip(keys, pars[2:]):
            subs[key] = d(val)
        text = (HEADER % subs) + (BODY % subs)
        path = os.path.join(base, name, 'bstrahl.f')
        with open(path, 'w') as fsock:
            fsock.write(text)
        print('wrote', path)

    for name in STUB_SETS:
        path = os.path.join(base, name, 'bstrahl.f')
        with open(path, 'w') as fsock:
            fsock.write(STUB_BODY % dict(name=name))
        print('wrote', path)


if __name__ == '__main__':
    main(sys.argv[1])
