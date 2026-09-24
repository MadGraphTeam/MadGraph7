      Subroutine transpole(pole1,width1,x1,y,jac)
c**********************************************************************
c     This routine transfers evenly spaced x values between 0 and 1
c     to y values with a pole at y=pole with width width and returns
c     the appropriate jacobian for this.  If x<del or x>1-del, uses
c     a linear transformation.  This ensures ability to cover entire 
c     region, even away from B.W.
c
c     If pole<0 then assumes have sqrt(1d0/(x^2+a^2)) type pole
c     If pole<0 then assumes have x/(x^2+a^2) type pole
c
c**********************************************************************
      implicit none
c
c     Constants
c
      double precision del
      parameter       (del=1d-22)          !Must agree with del in untranspole
c
c     Arguments
c
      double precision pole,width,y,jac
      double precision x1

c
c     Local
c
      double precision z,zmin,zmax,xmin,xmax,ez
      double precision pole1,width1,x,xc
      double precision a,b
c
c     small width treatment
c
      double precision small_width_treatment
      common/narrow_width/small_width_treatment
c-----
c  Begin Code
c-----
      pole=pole1
      width=width1

      x = x1
      if (pole .gt. 0d0) then
         if (width.lt.pole*small_width_treatment)then
            width = pole * small_width_treatment
            jac = jac * width/width1
         endif

         zmin = atan((-pole)/width)/width
         zmax = atan((1d0-pole)/width)/width
         if (x .gt. del .and. x .lt. 1d0-del) then
            z = zmin+(zmax-zmin)*x
            y = pole+width*tan(width*z)
            jac = jac *(width/cos(width*z))**2*(zmax-zmin)
         elseif (x .lt. del) then
            xmin = 0d0
            z    = zmin+(zmax-zmin)*del
            xmax = pole+width*tan(width*z)
            y = xmin+x*(xmax-xmin)/del
            jac = jac*(xmax-xmin)/del
         else
            xmax = 1d0
            z    = zmin+(zmax-zmin)*(1d0-del)
            xmin = pole+width*tan(width*z)
            y = xmin+(x+del-1d0)*(xmax-xmin)/del
            jac = jac*(xmax-xmin)/del
         endif
      elseif(pole .gt. -1d0) then       !1/sqrt(x^2+width^2) t-channel
         if (x .gt. .5d0) then          !Don't do anything here t>0
            y=x
         else
            zmin = log(2d0*width)       !2*width is because x->1-2*x
            zmax = log(1d0+sqrt(1d0+4d0*width*width))
            x=1d0-x*2d0
            z = zmin+(zmax-zmin)*x
            ez = exp(z)
            y = (1d0-.5d0*(ez-4d0*width*width/ez))/2d0
            jac = jac *(zmax-zmin)*.5d0*(ez+4d0*width*width/ez)
c            x = .5d0*(1d0-x)
         endif
c-------
c    tjs 3/5/2011  Perform 1/x transformation  using y=xo^(1-x)
c-------
      elseif(pole .eq. -15d0 .and. width .gt. 0d0) then !1/x   limit of width         
c         if (x .lt. width) then      !No transformation below cutoff
         xc = width
         xc = 1d0/(1d0-log(width))
         if (x .le. xc) then      !No transformation below cutoff
            y=x*width/xc
            jac = jac * width / xc
         else
            z = (x-xc)/(1d0-xc)
            y=width**(1d0-z)
            jac = jac * y * (-log(width))/(1d0-xc)
c            write(*,*) "trans",x,y,z
         endif
c         write(*,*) 'Transpole called',x,y
         return
      elseif(pole .ge. -2d0 .and. width .gt. 0d0) then !1/x^2   limit of width
         if (x .lt. width) then      !No transformation below cutoff
            y=x
         else
c---------
c   tjs 5/1/2008  modified for any y=x^-n transformation       
c-----------
            z = 1d0 - x + width
            b = ( 1d0-width) / (width**(pole+1d0) - 1d0)
            a = width - b
            y = a + b * z**(pole+1)
            jac = jac * abs((pole+1d0) * b * z**(pole))
c            write(*,*) "pre-trans",x,y
c            call untranspole(pole,width,x,y,jac)
c            write(*,*) "post-trans",x,y
c-----uncomment for 1/x^2 tjs -------
c            x = 1d0-x+width
c            y=width/x
c            jac = jac*width/(x*x)
c------------------------------------
            

c            write(*,*) 'trans',x,width/(x*x)
         endif

      elseif(pole .gt. -1d99) then       !1/sqrt(x^2+width^2) s-channel
         zmin = log(width)
         zmax = log(1d0+sqrt(1d0+width*width))
         if (x .gt. del .and. x .lt. 1d0-del) then
            z = zmin+(zmax-zmin)*x
            ez = exp(z)
            y = .5d0*(ez-width*width/ez)
            jac = jac *(zmax-zmin)*.5d0*(ez+width*width/ez)
         elseif (x .le. del) then
            xmin = 0d0
            z    = zmin+(zmax-zmin)*del
            ez   = exp(z)
            xmax = .5d0*(ez-width*width/ez)
            y = xmin+x*(xmax-xmin)/del
            jac = jac*(xmax-xmin)/del
         else
            xmax = 1d0
            z    = zmin+(zmax-zmin)*(1d0-del)
            ez   = exp(z)
            xmin = .5d0*(ez-width*width/ez)
            y = xmin+(x+del-1d0)*(xmax-xmin)/del
            jac = jac*(xmax-xmin)/del
         endif
      elseif(pole .gt. -8d99) then
         zmin = .5d0*log(width*width)
         zmax = .5d0*log(1d0+width*width)
         if (x .gt. del .and. x .lt. 1d0-del) then
            z = zmin+(zmax-zmin)*x
            ez = exp(2d0*z)
            y = sqrt(ez-width*width)
            jac = jac *(zmax-zmin)*ez/sqrt(ez-width*width)
         elseif (x .lt. del) then
            xmin = 0d0
            z    = zmin+(zmax-zmin)*del
            xmax = sqrt(exp(2d0*z)-width*width)
            y = xmin+x*(xmax-xmin)/del
            jac = jac*(xmax-xmin)/del
         else
            xmax = 1d0
            z    = zmin+(zmax-zmin)*(1d0-del)
            xmin = sqrt(exp(2d0*z)-width*width)
            y = xmin+(x+del-1d0)*(xmax-xmin)/del
            jac = jac*(xmax-xmin)/del
         endif
      endif
      end

      Subroutine untranspole(pole1,width1,x,y1,jac)
c**********************************************************************
c     This routine transfers takes values of y for a given pole and
c     width, and returns the value of x (which an evenly placed
c     random number) would have been used to get that value of y.
c     it also returns the jacobian associated with this choice.
c**********************************************************************
      implicit none
c
c     Constants
c
      double precision del
      parameter       (del=1d-22)          !Must agree with del in untranspole
c
c     Arguments
c
      double precision pole1,width1,y1,jac
      real*8 x
c
c     small width treatment
c
      double precision small_width_treatment
      common/narrow_width/small_width_treatment
c
c     Local
c
      double precision z,zmin,zmax,xmin,xmax,ez
      double precision pole,width,y,xc
      double precision a,b
      double precision xgmin,xgmax       ! these should be identical 
      parameter (xgmin=-1d0, xgmax=1d0)  ! to the ones in genps.inc
c-----
c  Begin Code
c-----
      pole=pole1
      width=width1
      y = y1
      if (pole .gt. 0d0) then                   !BW 
         if (width.lt.pole*small_width_treatment)then
            width = pole * small_width_treatment
            jac = jac * width/width1
         endif
         zmin = atan((-pole)/width)/width
         zmax = atan((1d0-pole)/width)/width
         z = atan((y-pole)/width)/width
         x = (z-zmin)/(zmax-zmin)
         if (x .le. del) then
            xmin = 0d0
            z    = zmin+(zmax-zmin)*del
            xmax = pole+width*tan(width*z)
            if(xmin.lt.xmax) then
               x = (y-xmin)*del/(xmax-xmin)
            else
               x=xmin
            endif
            jac = jac*(xmax-xmin)/del
         elseif (x .ge. 1d0-del) then
            xmax = 1d0
            z    = zmin+(zmax-zmin)*(1d0-del)
            xmin = pole+width*tan(width*z)
            if(xmin.lt.xmax) then
               x = (y-xmin)*del/(xmax-xmin)-del+1d0
            else
               x=xmin
            endif
            jac = jac*(xmax-xmin)/del
c RF (2014/07/07): code is not protected against this special case. In this case,
c simply set x to 1 and the jac to zero so that this PS point will not
c contribute (but you do get the correct xbin_min and xbin_max in
c sample_get_x)
            if (y.eq.xgmax .and. xmin.ge.xgmax) then
               x=1d0
               jac=0d0
            endif
         else
            jac = jac *(width/cos(width*z))**2*(zmax-zmin)
         endif
c-------
c    tjs 3/5/2011  Perform 1/x transformation  using y=xo^(1-x)
c-------
      elseif(pole .eq. -15d0 .and. width .gt. 0d0) then !1/x   limit of width
         xc = 1d0/(1d0-log(width))
c         xc = width
         if (y .le. width) then      !No transformation below cutoff
            x = y*xc/width
         else
            z = 1d0-log(y)/log(width)
            x = z*(1d0-xc) + xc
c            write(*,*) "untrans",x,y,z
         endif
         return
      elseif(pole .gt. -1d0) then !1/sqrt((.5-x)^2+width^2)  t-channel
         if (y .gt. .5d0) then
            x=y
         else
            zmin = log(width*2d0)
            zmax = log(1d0+sqrt(1d0+4d0*width*width))
            y = (1d0-2d0*y)
            z = log(y+sqrt(y*y+4d0*width*width))
            x = (z - zmin)/(zmax-zmin)
            x = .5d0*(1d0-x)
            ez = exp(z)
            jac = jac *(zmax-zmin)*.5d0*(ez+4d0*width*width/ez)
            y = (1d0-y)/2d0
         endif

      elseif(pole .gt. -5d0 .and. width .gt. 0d0) then !1/x^2   limit of width
         if (y .lt. width) then      !No transformation below cutoff
            x=y
         else
c---------
c   tjs 5/1/2008  modified for any y=x^-n transformation       
c-----------
            b = ( 1d0-width) / (width**(pole+1d0) - 1d0)
            a = width - b
            z = ((y-a)/b)**(1d0/(pole+1)) 
            x = 1d0 - z + width
            jac = jac * abs((pole+1d0) * b * z**(pole))

c-------------------
c Uncomment below for y=1/x^2
c-------------------
c            x=width/y
c            write(*,*) 'untr',x,width/(x*x)
c            jac = jac*width/(x*x)
c            x = 1d0-x+width
         endif

      elseif(pole .gt. -5d99) then !1/sqrt(x^2+width^2)  s-channel
         zmin = log(width)
         zmax = log(1d0+sqrt(1d0+width*width))
         if (pole .gt. -1d0 .and. y .lt. -pole) y=-pole-y
         z = log(y+sqrt(y*y+width*width))
         x = (z - zmin)/(zmax-zmin)
         if (x .gt. del .and. x .lt. 1d0-del) then
            ez = exp(z)
            jac = jac *(zmax-zmin)*.5d0*(ez+width*width/ez)
         elseif (x .lt. del) then
            xmin = 0d0
            z    = zmin+(zmax-zmin)*del
            ez   = exp(z)
            xmax = .5d0*(ez-width*width/ez)
c            y = xmin+x*(xmax-xmin)/del
            if(xmin.lt.xmax) then
               x = (y-xmin)*del/(xmax-xmin)
            else
               x=xmin
            endif
            jac = jac*(xmax-xmin)/del
         else
            xmax = 1d0
            z    = zmin+(zmax-zmin)*(1d0-del)
            ez   = exp(z)
            xmin = .5d0*(ez-width*width/ez)
c            y = xmin+(x+del-1d0)*(xmax-xmin)/del
            x = (y-xmin)*del/(xmax-xmin)-del+1d0
            jac = jac*(xmax-xmin)/del
         endif
      endif
      end


      subroutine bw_window_cdf(pole,width,wlo,whi,a0,f1,f2,tot,c)
c**********************************************************************
c     Normalisation of the flat-window Breit-Wigner density on [0,1]:
c     g(y) = 1/((y-pole)^2+width^2) outside [wlo,whi] and the constant
c     c = mean of g at the two window edges inside.
c     F(y) is its primitive with F(0)=0: f1=F(wlo), f2=F(whi), tot=F(1).
c     a0 = atan(-pole/width)/width is the offset of the arctan primitive.
c**********************************************************************
      implicit none
      double precision pole,width,wlo,whi,a0,f1,f2,tot,c
      double precision ga,gb
      a0 = atan(-pole/width)/width
      ga = 1d0/((wlo-pole)**2+width**2)
      gb = 1d0/((whi-pole)**2+width**2)
      c = 0.5d0*(ga+gb)
      f1 = atan((wlo-pole)/width)/width - a0
      f2 = f1 + c*(whi-wlo)
      tot = f2 + atan((1d0-pole)/width)/width
     &     - atan((whi-pole)/width)/width
      end


      subroutine transpole_win(pole1,width1,wlo1,whi1,x,y,jac)
c**********************************************************************
c     As transpole for a B.W. (pole>0) but for a $-excluded propagator:
c     the density is the B.W. outside the window [wlo,whi] (in the same
c     s/stot units as pole) and flat inside, at the mean of the two edge
c     values, so that the excluded pole region is not oversampled.
c     Same scheme as madspace's flat_window_breit_wigner_invariant.
c**********************************************************************
      implicit none
      double precision pole1,width1,wlo1,whi1,x,y,jac
      double precision pole,width,wlo,whi,a0,f1,f2,tot,c,u,g
      double precision small_width_treatment
      common/narrow_width/small_width_treatment

      pole = pole1
      width = width1
      if (width.lt.pole*small_width_treatment) then
         width = pole*small_width_treatment
         jac = jac*width/width1
      endif
      wlo = max(wlo1,0d0)
      whi = min(whi1,1d0)
      if (wlo.ge.whi) then
         call transpole(pole1,width1,x,y,jac)
         return
      endif
      call bw_window_cdf(pole,width,wlo,whi,a0,f1,f2,tot,c)
      u = x*tot
      if (u.lt.f1) then
         y = pole + width*tan(width*(u+a0))
         g = 1d0/((y-pole)**2+width**2)
      elseif (u.lt.f2) then
         y = wlo + (u-f1)/c
         g = c
      else
         y = pole + width*tan(width*(u-f2+atan((whi-pole)/width)/width))
         g = 1d0/((y-pole)**2+width**2)
      endif
      jac = jac*tot/g
      end


      subroutine untranspole_win(pole1,width1,wlo1,whi1,x,y,jac)
c**********************************************************************
c     Inverse of transpole_win: returns the x giving y, and multiplies
c     jac by dy/dx.
c**********************************************************************
      implicit none
      double precision pole1,width1,wlo1,whi1,x,y,jac
      double precision pole,width,wlo,whi,a0,f1,f2,tot,c,g
      double precision small_width_treatment
      common/narrow_width/small_width_treatment

      pole = pole1
      width = width1
      if (width.lt.pole*small_width_treatment) then
         width = pole*small_width_treatment
         jac = jac*width/width1
      endif
      wlo = max(wlo1,0d0)
      whi = min(whi1,1d0)
      if (wlo.ge.whi) then
         call untranspole(pole1,width1,x,y,jac)
         return
      endif
      call bw_window_cdf(pole,width,wlo,whi,a0,f1,f2,tot,c)
      if (y.lt.wlo) then
         x = (atan((y-pole)/width)/width - a0)/tot
         g = 1d0/((y-pole)**2+width**2)
      elseif (y.le.whi) then
         x = (f1 + c*(y-wlo))/tot
         g = c
      else
         x = (f2 + atan((y-pole)/width)/width
     &        - atan((whi-pole)/width)/width)/tot
         g = 1d0/((y-pole)**2+width**2)
      endif
      jac = jac*tot/g
      end
