      program test_bstrahl
c     Closure test: the normalisation of the sampling integrand in
c     bstrahl.f must reproduce Iapq() of vendor/eepdfgrid, and the
c     sampled split must satisfy xbs*xisr = x with xisr in [x,1].
      implicit none
      integer ngrid
      parameter (ngrid=128)
      double precision me, alphaem, ra
      parameter (me=0.511d-3)
      parameter (alphaem=0.007562397d0)
      parameter (ra=0.5d0)

      double precision xlist(5), q2list(3)
      data xlist /0.01d0, 0.2d0, 0.5d0, 0.9d0, 0.999d0/
      data q2list /100d0, 8315d0, 250000d0/

      double precision x, Q2, a, p, q, beta, pa, pb, rb, pi, h
      double precision mine, ref, f0, f1, tot1, tot2
      double precision Iapq, bstrahl_integrand
      external Iapq, bstrahl_integrand
      logical has
      integer i, j, k, n, beamid, nb
      integer beams(2)
      data beams /11, -11/
      double precision xbs, xisr, u1, u2, sbs, worst
      integer nsamp
      parameter (nsamp=20000)
      integer iseed
      double precision rnd

      pi = 4d0*datan(1d0)
      pa = -2d0/3d0
      rb = 1d0-ra
      h = 1d0/dble(ngrid)
      worst = 0d0

      write(*,*) '--- normalisation closure vs Iapq ---'
      do nb = 1, 2
        beamid = beams(nb)
        do n = 1, 4
          call eepdf_bstrahl_pars(n,beamid,beamid,a,p,q,has)
          if (.not.has) cycle
          do j = 1, 3
            Q2 = q2list(j)
            beta = alphaem/pi*(dlog(Q2/me/me)-1d0)
            pb = beta-1d0
            do i = 1, 5
              x = xlist(i)
              tot1 = 0d0
              f0 = bstrahl_integrand(0d0,1,x,p,q,beta,pa,pb,ra,rb)
              do k = 1, ngrid
                f1 = bstrahl_integrand(dble(k)*h,1,x,p,q,beta,pa,pb,
     &                                 ra,rb)
                tot1 = tot1 + 0.5d0*h*(f0+f1)
                f0 = f1
              enddo
              tot2 = 0d0
              f0 = bstrahl_integrand(0d0,2,x,p,q,beta,pa,pb,ra,rb)
              do k = 1, ngrid
                f1 = bstrahl_integrand(dble(k)*h,2,x,p,q,beta,pa,pb,
     &                                 ra,rb)
                tot2 = tot2 + 0.5d0*h*(f0+f1)
                f0 = f1
              enddo
              mine = a*(tot1+tot2)
              ref = Iapq(x,a,p,q,beta)
              if (dabs(ref).gt.0d0) then
                if (dabs(mine/ref-1d0).gt.worst)
     &            worst = dabs(mine/ref-1d0)
                write(*,'(a,i2,a,i3,a,f8.4,a,f10.1,3(a,e14.6))')
     &            ' n=',n,' beam=',beamid,' x=',x,' Q2=',Q2,
     &            '  mine=',mine,' Iapq=',ref,' rel=',mine/ref-1d0
              endif
            enddo
          enddo
        enddo
      enddo
      write(*,*) 'worst relative deviation:', worst

      write(*,*) '--- sampling sanity (n=4, e- beam) ---'
      iseed = 12345
      Q2 = 8315d0
      do i = 1, 5
        x = xlist(i)
        sbs = 0d0
        do k = 1, nsamp
          call ranmar(iseed,u1)
          call ranmar(iseed,u2)
          call eepdf_sample_bstrahl(x,Q2,4,11,11,u1,u2,xbs,xisr)
          if (dabs(xbs*xisr/x-1d0).gt.1d-10) then
            write(*,*) 'FAIL: xbs*xisr != x', x, xbs, xisr
            stop 1
          endif
          if (xisr.lt.x-1d-12 .or. xisr.gt.1d0+1d-12) then
            write(*,*) 'FAIL: xisr out of [x,1]', x, xisr
            stop 1
          endif
          sbs = sbs + xbs
        enddo
        write(*,'(a,f8.4,a,f10.6)') ' x=',x,'  <xbs>=',sbs/dble(nsamp)
      enddo

      end


      subroutine ranmar(iseed,r)
c     minimal LCG, good enough for a sanity test
      implicit none
      integer iseed
      double precision r
      iseed = mod(16807*iseed, 2147483647)
      if (iseed.le.0) iseed = iseed + 2147483646
      r = dble(iseed)/2147483647d0
      return
      end
