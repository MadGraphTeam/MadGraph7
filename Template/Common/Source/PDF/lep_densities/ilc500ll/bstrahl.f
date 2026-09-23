c     ------------------------------------------------------------------
c     Beamstrahlung / ISR splitting for the 'ilc500ll' lepton density.
c
c     eepdf.f ships phibar(x,Q2), the Mellin convolution of a CIRCE
c     beamstrahlung spectrum with the ISR structure function.  Writing
c     z for the ISR momentum fraction and t = (z-x)/(1-x), the reduced
c     density of a beamstrahlung component is
c
c        phibar(x,Q2) = int_0^1 dt  a * exp(p*u) * dexp(q*u**1.5d0)
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
c     Generated from vendor/eepdfgrid/calcpdf_ilc500.f by
c     gen_bstrahl.py -- do not edit by hand.
c     ------------------------------------------------------------------

      logical function eepdf_has_bstrahl()
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
      data a11/0.5012d0/,a01/0.1613d0/,a10/0.1613d0/,a00/0.06379d0/
      data p1/-8.514d0/,q1/-5.808d0/,p2/-8.505d0/,q2/-5.823d0/
      data p3/-7.535d0/,q3/-6.79d0/,p4/-7.481d0/,q4/-6.849d0/

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
c     Gmu scheme, as in gridpdfaux.f / calcpdf_ilc500.f
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

      g = dexp(p*u)*dexp(q*u**1.5d0)*bstrahl_isr_tilde(z,beta)
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
