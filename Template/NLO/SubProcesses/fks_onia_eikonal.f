c**************************************************************************
c     Eikonal tensor integrals of the integrated soft counterterms of a
c     P-wave quarkonium (arXiv:2402.19221, appendix A), in the conventions
c     of eikonal_Ireg (fks_singular.f): momenta in the partonic c.m. frame,
c     shat, QES2 = Q_ES^2, xicut. Each integral is returned as the
c     coefficients P2 (1/eps^2) and P1 (1/eps) in front of
c     (4 pi)^eps/Gamma(1-eps) (mu^2/Q_ES^2)^eps, and the finite part FIN.
c     Vectors and tensors carry upper indices. Checked against
c     finite-difference derivatives of the scalar integrals
c     (onia_fks/docs/scripts/eikonal.py).
c**************************************************************************

      double precision function onia_dot(a,b)
      implicit none
      double precision a(0:3),b(0:3)
      onia_dot=a(0)*b(0)-a(1)*b(1)-a(2)*b(2)-a(3)*b(3)
      end


      double precision function onia_l1p(z)
c     log(1+z), accurate for small |z| (Goldberg: log(u) z/(u-1), u = 1+z)
      implicit none
      double precision z,u
      u=1d0+z
      if (u.eq.1d0) then
         onia_l1p=z
      else
         onia_l1p=dlog(u)*z/(u-1d0)
      endif
      end


      double precision function onia_l1pz(z)
c     log(1+z)/z, 1 at z = 0
      implicit none
      double precision z,onia_l1p
      external onia_l1p
      if (z.eq.0d0) then
         onia_l1pz=1d0
      else
         onia_l1pz=onia_l1p(z)/z
      endif
      end


      subroutine onia_eik_r1_self(pl,shat,qes2,xicut,p1,fin)
c**************************************************************************
c     E-bar^mu({1,2},{l,l}) = l^2 int q^mu/(l.q)^3, l massive: the 1/eps
c     coefficient P1 (no 1/eps^2 pole) and the finite part FIN.
c**************************************************************************
      implicit none
      double precision pl(0:3),shat,qes2,xicut,p1(0:3),fin(0:3)
      double precision onia_dot,onia_gself
      external onia_dot,onia_gself
      double precision d0(0:3),m2,e,b,ll
      data d0 /1d0,0d0,0d0,0d0/
      ll=dlog(xicut**2*shat/qes2)
      m2=onia_dot(pl,pl)
      e=pl(0)
      b=dsqrt(pl(1)**2+pl(2)**2+pl(3)**2)/e
      p1(0:3)=-pl(0:3)/m2
      fin(0:3)=(onia_gself(1,b)+2d0*ll)/(2d0*m2)*pl(0:3)
     &     +onia_gself(2,b)/(2d0*e)*d0(0:3)
      end


      subroutine onia_eik_r1(pk,pl,lself,shat,qes2,xicut,p2,p1,fin)
c**************************************************************************
c     E-bar^mu({1,2},{k,l}) = k.l int q^mu/((k.q)(l.q)^2): l massive
c     (power 2), k massless, massive, or l itself (lself). The mass of a
c     momentum is taken from its square.
c     The closed forms of appendix A have removable singularities when k
c     and l are collinear in the c.m. frame, which is always the case for
c     the two final-state particles of a 2 -> 2 Born (back to back): there
c     the massive-massive form gives NaN and the massless-massive one a
c     wrong finite part. The massless-massive form is evaluated here with
c     x-a and x-c from the angle between k and l and log(1+z)/z; the
c     massive-massive case uses Feynman parameters,
c       1/((k.q)(l.q)^2) = int_0^1 dx 2(1-x)/(P_x.q)^3, P_x = x k+(1-x) l,
c       E^mu({1,2},{k,l}) = k.l int_0^1 dx 2(1-x) E^mu({1,2},{P_x,P_x})/P_x^2,
c     with the self form and a 64-point Gauss-Legendre rule (smooth
c     integrand, P_x timelike); it equals the closed form to 1d-13 away
c     from collinearity. A massless k exactly parallel to l (za -> 0, not
c     a configuration of the final state of a 2 -> 2 Born) keeps a
c     cancellation of ~1d-16/za, as the closed form.
c**************************************************************************
      implicit none
      double precision pk(0:3),pl(0:3),shat,qes2,xicut
      logical lself
      double precision p2(0:3),p1(0:3),fin(0:3)
      double precision onia_dot,ddilog,pi,onia_l1p,onia_l1pz
      external onia_dot,ddilog,onia_l1p,onia_l1pz
      parameter (pi=3.1415926535897932385d0)
      integer ngl
      parameter (ngl=64)
      double precision xgl(ngl),wgl(ngl)
      logical firsttime
      data firsttime /.true./
      save firsttime,xgl,wgl
      double precision d0(0:3),bl,el,ek,x,mk2,ml2,ml,a,c,lbb,t1,t2,t3,b
      double precision ll,pp(0:3),p1s(0:3),fins(0:3),wt,pkn,pln,sm,sp
      double precision za,zc,lxa,lxc,fa,fc
      integer i
      data d0 /1d0,0d0,0d0,0d0/
      ll=dlog(xicut**2*shat/qes2)
      p2(0:3)=0d0
      if (lself) then
         call onia_eik_r1_self(pl,shat,qes2,xicut,p1,fin)
         return
      endif
      mk2=onia_dot(pk,pk)
      ml2=onia_dot(pl,pl)
      x=onia_dot(pk,pl)
      ek=pk(0)
      el=pl(0)
      bl=dsqrt(pl(1)**2+pl(2)**2+pl(3)**2)/el
      if (dabs(mk2).lt.1d-10*ek**2) then
c        massless k
         ml=dsqrt(ml2)
         b=bl
         p2(0:3)=pk(0:3)/(2d0*x)
         p1(0:3)=(1d0-dlog(x/(ek*ml))-ll/2d0)/x*pk(0:3)-pl(0:3)/ml2
         if (b.lt.3d-6) then
c           l (nearly) at rest, as the onium of a 2 -> 1 Born: the limit
c           b -> 0, independent of the direction (the closed form below
c           loses ~1d-16/b^2; the limit is off by O(b))
            fin(0:3)=(2d0-pi**2/12d0-ll+ll**2/4d0)/x*pk(0:3)
     &           +(ll-2d0)/ml2*pl(0:3)
            return
         endif
         a=ek*el*(1d0-b)
         c=ek*el*(1d0+b)
         lbb=dlog((1d0+b)/(1d0-b))
c        za = (x-a)/a = ek el b (1-cos)/a, zc = (x-c)/c = -ek el b (1+cos)/c,
c        1 -/+ cos = |k^ -/+ l^|^2/2 from the unit vectors (accurate when k
c        and l are back to back, as a parton recoiling against the onium)
         pkn=dsqrt(pk(1)**2+pk(2)**2+pk(3)**2)
         pln=dsqrt(pl(1)**2+pl(2)**2+pl(3)**2)
         sm=0d0
         sp=0d0
         do i=1,3
            sm=sm+(pk(i)/pkn-pl(i)/pln)**2
            sp=sp+(pk(i)/pkn+pl(i)/pln)**2
         enddo
         za=ek*el*b*sm/(2d0*a)
         zc=-ek*el*b*sp/(2d0*c)
         lxa=onia_l1p(za)
         lxc=onia_l1p(zc)
         fa=onia_l1pz(za)
         fc=onia_l1pz(zc)
c        x/(x-a) log(x/a) = (1+za) fa, c/(x-c) log(x/c) = fc,
c        ek el/(x-a) log(x/a) = fa/(1-b), ek el/(x-c) log(x/c) = fc/(1+b),
c        and [ek el (log(c/x)-log(x/a)) + x lbb/(1+b)]/(x-c)
c          = (lbb - 2 fc)/(1+b)   (log(x/a) = lbb + log(x/c))
         t1=-pi**2/12d0-ll+ll**2/4d0+dlog(x/(ek*ml))*ll
     &        -lxa+lxa**2/2d0-lbb**2/4d0
     &        +ddilog(-za)-ddilog(zc/(1d0+zc))
     &        +(1d0+za)*fa+fc
         t2=ll+ml2/(b*el**2)*(-lbb/(1d0-b**2)
     &        -fa/(1d0-b)+fc/(1d0+b))
         t3=(-ek*(lxc+lxa)
     &        +x*ml2/(el**3*b*(1d0-b**2))*lbb
     &        +ek*(el**2*b*(1d0+b)+ml2)/el**2*(lbb-2d0*fc)/(1d0+b))
     &        /(a*za)
         fin(0:3)=t1/x*pk(0:3)+t2/ml2*pl(0:3)+t3*d0(0:3)
         return
      endif
c     two massive momenta: Feynman parameters
      if (firsttime) then
         call onia_gauleg(ngl,xgl,wgl)
         firsttime=.false.
      endif
      p1(0:3)=0d0
      fin(0:3)=0d0
      do i=1,ngl
         pp(0:3)=xgl(i)*pk(0:3)+(1d0-xgl(i))*pl(0:3)
         call onia_eik_r1_self(pp,shat,qes2,xicut,p1s,fins)
         wt=2d0*(1d0-xgl(i))*wgl(i)*x/onia_dot(pp,pp)
         p1(0:3)=p1(0:3)+wt*p1s(0:3)
         fin(0:3)=fin(0:3)+wt*fins(0:3)
      enddo
      end


      subroutine onia_eik_r2_self(pk,shat,qes2,xicut,p1,fin)
c**************************************************************************
c     E-bar^{mu nu}({2,2},{k,k}), k massive (no 1/eps^2 pole)
c**************************************************************************
      implicit none
      double precision pk(0:3),shat,qes2,xicut,p1(0:3,0:3),fin(0:3,0:3)
      double precision onia_dot,onia_gself
      external onia_dot,onia_gself
      double precision m2,e,b,ll,c1,c2,c3,c4,g(0:3),d0(0:3)
      integer mu,nu
      data g /1d0,-1d0,-1d0,-1d0/
      data d0 /1d0,0d0,0d0,0d0/
      ll=dlog(xicut**2*shat/qes2)
      m2=onia_dot(pk,pk)
      e=pk(0)
      b=dsqrt(pk(1)**2+pk(2)**2+pk(3)**2)/e
      c1=(1d0-b**2)*onia_gself(3,b)/(6d0*m2)
      c2=(onia_gself(4,b)-2d0*ll)/(6d0*m2)
      c3=onia_gself(5,b)/(6d0*e*m2)
      c4=(onia_gself(6,b)+8d0*ll)/(6d0*m2**2)
      do mu=0,3
         do nu=0,3
            p1(mu,nu)=-4d0*pk(mu)*pk(nu)/(3d0*m2**2)
            fin(mu,nu)=c1*d0(mu)*d0(nu)
     &           +c3*(d0(mu)*pk(nu)+pk(mu)*d0(nu))+c4*pk(mu)*pk(nu)
            if (mu.eq.nu) then
               p1(mu,nu)=p1(mu,nu)+g(mu)/(3d0*m2)
               fin(mu,nu)=fin(mu,nu)+c2*g(mu)
            endif
         enddo
      enddo
      end


      double precision function onia_gself(ig,b)
c**************************************************************************
c     The velocity functions of the massive self-eikonal integrals, with
c     lb = log((1+b)/(1-b)), b the velocity of the onium:
c       1  (lb - 2b - 3 lb b^2)/b^3
c       2  (-lb + 2b + lb b^2)/b^3
c       3  (-3 lb + 6b + 3 lb b^2 - 4b^3)/b^5
c       4  (-lb + 2b + 3 lb b^2)/b^3
c       5  (3 lb - 6b - 6 lb b^2 + 10b^3 + 3 lb b^4)/b^5
c       6  (-3 lb + 6b + 10 lb b^2 - 18b^3 - 15 lb b^4)/b^5
c     For b < 0.2 (an onium at rest in 2 -> 1 Borns: b = 0) the exact
c     series in b^2 from lb = 2 sum b^(2k+1)/(2k+1), 14 terms (< 1d-20);
c     above, the closed forms (cancellation < 1d-16/b^5).
c**************************************************************************
      implicit none
      integer ig,n
      double precision b,lb,b2,bn,t
      if (b.lt.0.2d0) then
         b2=b**2
         bn=1d0
         onia_gself=0d0
         do n=0,13
            if (ig.eq.1) then
               t=2d0/(2*n+3)-6d0/(2*n+1)
            elseif (ig.eq.2) then
               t=-2d0/(2*n+3)+2d0/(2*n+1)
            elseif (ig.eq.3) then
               t=-6d0/(2*n+5)+6d0/(2*n+3)
            elseif (ig.eq.4) then
               t=-2d0/(2*n+3)+6d0/(2*n+1)
            elseif (ig.eq.5) then
               t=6d0/(2*n+5)-12d0/(2*n+3)+6d0/(2*n+1)
            else
               t=-6d0/(2*n+5)+20d0/(2*n+3)-30d0/(2*n+1)
            endif
            onia_gself=onia_gself+t*bn
            bn=bn*b2
         enddo
         return
      endif
      lb=dlog((1d0+b)/(1d0-b))
      if (ig.eq.1) then
         onia_gself=(lb-2d0*b-3d0*lb*b**2)/b**3
      elseif (ig.eq.2) then
         onia_gself=(-lb+2d0*b+lb*b**2)/b**3
      elseif (ig.eq.3) then
         onia_gself=(-3d0*lb+6d0*b+3d0*lb*b**2-4d0*b**3)/b**5
      elseif (ig.eq.4) then
         onia_gself=(-lb+2d0*b+3d0*lb*b**2)/b**3
      elseif (ig.eq.5) then
         onia_gself=(3d0*lb-6d0*b-6d0*lb*b**2+10d0*b**3+3d0*lb*b**4)
     &        /b**5
      else
         onia_gself=(-3d0*lb+6d0*b+10d0*lb*b**2-18d0*b**3
     &        -15d0*lb*b**4)/b**5
      endif
      end


      subroutine onia_eik_dd(k1,k2,e11,shat,qes2,xicut,p1,fin)
c**************************************************************************
c     The dipole-dipole tensor of two massive momenta (JHEP_2Pwave,
c     eq. Dbar), upper indices,
c       Dbar^{mu nu}(K1,K2) = g^{mu nu}/(K1.K2) E({1,1},{K1,K2})
c                - K2^mu/(K1.K2) E^nu({1,2},{K1,K2})
c                - K1^nu/(K1.K2) E^mu({1,2},{K2,K1})
c                + E^{mu nu}({2,2},{K1,K2})
c                = d^2 E({1,1},{K1,K2})/dK1_mu dK2_nu,
c     mu belonging to K1. E11 is the finite part of E({1,1},{K1,K2})
c     (eikonal_Ireg), the rank-1 integrals are the closed forms of
c     onia_eik_r1, and the rank 2 the Feynman-parameter representation
c     (eq. E22fin)
c       E^{mu nu}({2,2},{K1,K2}) = 6 K1.K2 int_0^1 dx x(1-x)/P_x^2
c                                  E^{mu nu}({2,2},{P_x,P_x}),
c     P_x = x K1 + (1-x) K2, with the massive self tensor onia_eik_r2_self
c     and a 64-point Gauss-Legendre rule (smooth integrand). P1 is the
c     coefficient of 1/eps (no 1/eps^2 pole), with that of E({1,1}) equal
c     to -W(omega), W(omega) = omega log(omega+sqrt(omega^2-1))
c     /sqrt(omega^2-1); FIN the finite part.
c**************************************************************************
      implicit none
      double precision k1(0:3),k2(0:3),e11,shat,qes2,xicut
      double precision p1(0:3,0:3),fin(0:3,0:3)
      integer ngl
      parameter (ngl=64)
      double precision xgl(ngl),wgl(ngl)
      logical firsttime
      data firsttime /.true./
      save firsttime,xgl,wgl
      double precision onia_dot
      external onia_dot
      double precision x12,m1,m2,omega,t,wom,pp(0:3),px2,wt,g(0:3)
      double precision p2a(0:3),p1a(0:3),fina(0:3)
      double precision p2b(0:3),p1b(0:3),finb(0:3)
      double precision t1(0:3,0:3),t2(0:3,0:3),e22p(0:3,0:3)
      double precision e22f(0:3,0:3)
      integer i,mu,nu
      data g /1d0,-1d0,-1d0,-1d0/
      if (firsttime) then
         call onia_gauleg(ngl,xgl,wgl)
         firsttime=.false.
      endif
      x12=onia_dot(k1,k2)
      m1=dsqrt(onia_dot(k1,k1))
      m2=dsqrt(onia_dot(k2,k2))
      omega=x12/(m1*m2)
      t=omega**2-1d0
      if (t.lt.1d-6) then
         wom=1d0+t/3d0-2d0*t**2/15d0
      else
         wom=omega*dlog(omega+dsqrt(t))/dsqrt(t)
      endif
      call onia_eik_r1(k1,k2,.false.,shat,qes2,xicut,p2a,p1a,fina)
      call onia_eik_r1(k2,k1,.false.,shat,qes2,xicut,p2b,p1b,finb)
      e22p(:,:)=0d0
      e22f(:,:)=0d0
      do i=1,ngl
         pp(0:3)=xgl(i)*k1(0:3)+(1d0-xgl(i))*k2(0:3)
         px2=onia_dot(pp,pp)
         call onia_eik_r2_self(pp,shat,qes2,xicut,t1,t2)
         wt=6d0*x12*wgl(i)*xgl(i)*(1d0-xgl(i))/px2
         e22p(:,:)=e22p(:,:)+wt*t1(:,:)
         e22f(:,:)=e22f(:,:)+wt*t2(:,:)
      enddo
      do mu=0,3
         do nu=0,3
            p1(mu,nu)=-k2(mu)*p1a(nu)/x12-k1(nu)*p1b(mu)/x12
     &           +e22p(mu,nu)
            fin(mu,nu)=-k2(mu)*fina(nu)/x12-k1(nu)*finb(mu)/x12
     &           +e22f(mu,nu)
            if (mu.eq.nu) then
               p1(mu,nu)=p1(mu,nu)-g(mu)*wom/x12
               fin(mu,nu)=fin(mu,nu)+g(mu)*e11/x12
            endif
         enddo
      enddo
      end


      subroutine onia_f12(omega,f1,f2,f1i,f2i)
c**************************************************************************
c     The dipole-dipole kernels of two P-wave pairs (JHEP_2Pwave,
c     eqs. f1f2, f1f2c; He, Kniehl and Wang, eq. 8), f1 = -W'/2 and
c     f2 = -W''/2 of omega = v1.v2, and the imaginary parts f1i, f2i of
c     f1c, f2c for two outgoing pairs (sigma = 1). For t = omega^2 - 1 <
c     0.1 the Taylor series in t (eq. f1f2series, 18 terms: the closed
c     forms cancel as t -> 0; f1(1) = -1/3, f2(1) = 1/5). The imaginary
c     parts (the Coulomb phase) are singular at t = 0 and are set to zero
c     there.
c**************************************************************************
      implicit none
      double precision omega,f1,f2,f1i,f2i
      double precision pi
      parameter (pi=3.1415926535897932385d0)
      double precision t,st,lg,ak,tk
      integer k
      t=omega**2-1d0
      if (t.lt.0.1d0) then
         f1=0d0
         ak=1d0
         tk=1d0
         do k=0,17
            if (k.gt.0) ak=ak*dble(2*k-1)/dble(2*k)
            f1=f1-ak*tk*(-1d0)**k/dble(2*k+3)
            tk=tk*t
         enddo
         f2=0d0
         ak=1d0
         tk=1d0
         do k=1,18
            ak=ak*dble(2*k-1)/dble(2*k)
            f2=f2+2d0*omega*dble(k)*ak*(-1d0)**(k+1)*tk/dble(2*k+3)
            tk=tk*t
         enddo
      else
         st=dsqrt(t)
         lg=dlog(omega+st)
         f1=(lg-omega*st)/(2d0*t*st)
         f2=((omega**2+2d0)*st-3d0*omega*lg)/(2d0*t**2*st)
      endif
      if (t.gt.0d0) then
         f1i=-pi/(2d0*t*dsqrt(t))
         f2i=3d0*pi*omega/(2d0*t**2*dsqrt(t))
      else
         f1i=0d0
         f2i=0d0
      endif
      end


      subroutine onia_gauleg(n,x,w)
c     the nodes and weights of the n-point Gauss-Legendre rule on [0,1]
      implicit none
      integer n,i,j,m,k
      double precision x(n),w(n),z,z1,p1,p2,p3,pp,pi
      parameter (pi=3.1415926535897932385d0)
      m=(n+1)/2
      do i=1,m
         z=dcos(pi*(i-0.25d0)/(n+0.5d0))
c        Newton iterations (quadratic convergence), and one more
         do k=1,100
            p1=1d0
            p2=0d0
            do j=1,n
               p3=p2
               p2=p1
               p1=((2d0*j-1d0)*z*p2-(j-1d0)*p3)/j
            enddo
            pp=n*(z*p1-p2)/(z*z-1d0)
            z1=z
            z=z1-p1/pp
            if (dabs(z-z1).lt.1d-14) goto 10
         enddo
 10      continue
         p1=1d0
         p2=0d0
         do j=1,n
            p3=p2
            p2=p1
            p1=((2d0*j-1d0)*z*p2-(j-1d0)*p3)/j
         enddo
         pp=n*(z*p1-p2)/(z*z-1d0)
         x(i)=0.5d0*(1d0-z)
         x(n+1-i)=0.5d0*(1d0+z)
         w(i)=1d0/((1d0-z*z)*pp*pp)
         w(n+1-i)=w(i)
      enddo
      end
