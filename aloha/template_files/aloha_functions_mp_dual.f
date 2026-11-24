c ======================================================================
c  Quadruple-precision dual external wavefunctions (MadLoop with one P-wave
c  bound state), generated from those of aloha_functions_dual.f by
c  tools/make_mp_dual_aloha_functions.py: mp_ names, type(MP_Dual) and
c  quadruple-precision types, constants and constructors.
c ======================================================================

      subroutine mp_ixxxxx(p,fmass,nhel,nsf,flavor,fi)
      use ALOHA_OBJECT
c
c This subroutine computes a fermion wavefunction derivative with the flowing-IN
c fermion number.
c
c input:
c       complex p(0:3)         : four-momentum of fermion
c       real    fmass          : mass          of fermion
c       integer nhel = -1 or 1 : helicity      of fermion
c       integer nsf  = -1 or 1 : +1 for particle, -1 for anti-particle
c
c output:
c       complex fi%W(4)         : fermion wavefunction derivative      eps_L^alpha . d|fi>/dq^alpha
c
      implicit none
      type(mp_aloha) fi
      integer flavor
      type(MP_Dual)::chi(2)
      type(MP_Dual)::p(0:3),omega(2),sfomeg(2),
     &            pp,pp3,sqp0p3
      real*16 sf(2),fmass,sqm(0:1),oHsqm(0:1)
      integer nhel,nsf,ip,im,nh
      integer i

      real*16 rZero, rHalf, rTwo
      complex*32 ci
      parameter( rZero = 0.0E0_16, rHalf = 0.5E0_16, rTwo = 2.0E0_16 )
      parameter( ci = cmplx(0.0E0_16,1.0E0_16,kind=16) )

c     Convention for dual computations (same as loop)
      fi%P(0) = p(0)*(-nsf)
      fi%P(1) = p(1)*(-nsf)
      fi%P(2) = p(2)*(-nsf)
      fi%P(3) = p(3)*(-nsf)
      do i = 1,4
         CALL fi%W(i)%initZERO()
      enddo
      fi%flv_index = flavor

      nh = nhel*nsf

      if (fmass.ne.rZero) then
         pp = sqrt(p(1)**2+p(2)**2+p(3)**2)
         pp%comp(0) = min(p(0)%comp(0)%re, pp%comp(0)%re)

         if (pp%comp(0)%re.eq.rZero) then
            ip = (1+nh)/2
            im = (1-nh)/2

            sqm(0) = sqrt(abs(fmass))  ! possibility of negative fermion masses
            sqm(1) = sign(sqm(0),fmass) ! possibility of negative fermion masses
            oHsqm(0) = rHalf/sqm(0)     ! normalization for the derivative part
            oHsqm(1) = rHalf/sqm(1)     ! normalization for the derivative part

            fi%W(1)%comp(0) = ip     * sqm(ip)
            fi%W(2)%comp(0) = im*nsf * sqm(ip)
            fi%W(3)%comp(0) = ip*nsf * sqm(im)
            fi%W(4)%comp(0) = im     * sqm(im)    
            do i = 1, size(p(0))
               fi%W(1)%comp(i) = oHsqm(ip)*
     &            (+ip*    (p(0)%comp(i) -    p(3)%comp(i))
     &             -im*nsf*(p(1)%comp(i) - ci*p(2)%comp(i))) 

               fi%W(2)%comp(i) = oHsqm(ip)*
     &            (-ip*    (p(1)%comp(i) + ci*p(2)%comp(i))
     &             +im*nsf*(p(0)%comp(i) +    p(3)%comp(i)))

               fi%W(3)%comp(i) = oHsqm(im)*
     &            (+ip*nsf*(p(0)%comp(i) +    p(3)%comp(i))
     &             +im*    (p(1)%comp(i) - ci*p(2)%comp(i))) 

               fi%W(4)%comp(i) = oHsqm(im)*
     &            (+ip*nsf*(p(1)%comp(i) + ci*p(2)%comp(i))
     &             +im*    (p(0)%comp(i) -    p(3)%comp(i)))
            enddo

         else
            ip = (3+nh)/2
            im = (3-nh)/2

            sf(1) = qreal(1+nsf+(1-nsf)*nh)*rHalf
            sf(2) = qreal(1+nsf-(1-nsf)*nh)*rHalf
            omega(1) = sqrt(p(0) + pp)
            omega(2) = fmass/omega(1)
            sfomeg(1) = sf(1)*omega(ip)
            sfomeg(2) = sf(2)*omega(im)

            pp3 = pp + p(3)
            pp3%comp(0) = max(pp3%comp(0)%re, rZero)
      
            if ((p(1)%comp(0)%re.eq.0d0).and.(p(2)%comp(0)%re.eq.0d0)
     &          .and.(p(3)%comp(0)%re.lt.0d0)) then
               oHsqm(0) = rHalf*sfomeg(1)%comp(0)/pp%comp(0)
               oHsqm(1) = rHalf*sfomeg(2)%comp(0)/pp%comp(0)
               chi(2)%comp(0) = qcmplx(-nh)
            else
               oHsqm(0:1) = (/0E0_16,0E0_16/)
               chi(1) = sqrt(rHalf*pp3/pp)
               chi(2) = (qreal(nh)*p(1) + ci*p(2))/(sqrt(rTwo*pp*pp3))
            endif

            fi%W(1) = sfomeg(1)*chi(im) 
     &            + (p(1) - nh*ci*p(2))*oHsqm(0)*chi(ip)
            fi%W(2) = sfomeg(1)*chi(ip) 
     &            - (p(1) - nh*ci*p(2))*oHsqm(0)*chi(im)
            fi%W(3) = sfomeg(2)*chi(im) 
     &            + (p(1) - nh*ci*p(2))*oHsqm(1)*chi(ip)
            fi%W(4) = sfomeg(2)*chi(ip) 
     &            - (p(1) - nh*ci*p(2))*oHsqm(1)*chi(im)
         endif

      else
c        sqp0p3 is 0 for a momentum along -z, as in HELAS, and so are its
c        derivatives where p0 + p3 = 0 (the dual square root of 0 has none)
         call sqp0p3%initZERO()
         if((p(1)%comp(0)%re.ne.0d0).or.(p(2)%comp(0)%re.ne.0d0).or.
     &      (p(3)%comp(0)%re.gt.0d0)) then
            if (p(0)%comp(0)%re+p(3)%comp(0)%re.gt.rZero)
     &         sqp0p3 = sqrt(p(0)+p(3))*nsf
            sqp0p3%comp(0) = sqrt(max(p(0)%comp(0)%re+
     &                        p(3)%comp(0)%re,rZero))*nsf
         end if
         chi(1) = sqp0p3
         if (sqp0p3%comp(0).eq.rZero) then
            chi(2) = qcmplx(-nhel)*sqrt(rTwo*p(0))
         else
            chi(2) = (nh*p(1) + ci*p(2))/sqp0p3
         endif
         if (nh.eq.1) then
            fi%W(3) = chi(1)
            fi%W(4) = chi(2)
         else
            fi%W(1) = chi(2)
            fi%W(2) = chi(1)
         endif
      endif
c
      return
      end subroutine mp_ixxxxx

      subroutine mp_oxxxxx(p,fmass,nhel,nsf,flavor,fo)
      use ALOHA_OBJECT
c
c This subroutine computes a fermion wavefunction derivative with the flowing-OUT
c fermion number.
c
c input:
c       complex p(0:3)         : four-momentum of fermion
c       real    fmass          : mass          of fermion
c       integer nhel = -1 or 1 : helicity      of fermion
c       integer nsf  = -1 or 1 : +1 for particle, -1 for anti-particle
c
c output:
c       complex fo%W(4)         : fermion wavefunction derivative       eps_L^alpha . d<fo|/dq^alpha
c
      implicit none
      type(mp_aloha) fo
      integer flavor
      type(MP_Dual)::chi(2)
      type(MP_Dual)::p(0:3),omega(2),sfomeg(2),
     &            pp,pp3,sqp0p3
      type(MP_Dual)::p3_tmp
      real*16 sf(2),fmass,sqm(0:1),oHsqm(0:1)
      integer nhel,nsf,ip,im,nh
      integer i

      real*16 rZero, rHalf, rTwo
      complex*32 ci
      parameter( rZero = 0.0E0_16, rHalf = 0.5E0_16, rTwo = 2.0E0_16 )
      parameter( ci = cmplx(0.0E0_16,1.0E0_16,kind=16) )

c     Convention for dual computations (same as loop)
      fo%P(0) = p(0)*(nsf)
      fo%P(1) = p(1)*(nsf)
      fo%P(2) = p(2)*(nsf)
      fo%P(3) = p(3)*(nsf)
      do i = 1,4
         CALL fo%W(i)%initZERO()
      enddo
      fo%flv_index = flavor

      nh = nhel*nsf

      if (fmass.ne.rZero) then
         pp = sqrt(p(1)**2+p(2)**2+p(3)**2)
         pp%comp(0) = min(p(0)%comp(0)%re, pp%comp(0)%re)

         if (pp%comp(0)%re.eq.rZero) then
            im = +nhel * (1+nh)/2
            ip = -nhel * (1-nh)/2

            sqm(0) = sqrt(abs(fmass))  ! possibility of negative fermion masses
            sqm(1) = sign(sqm(0),fmass) ! possibility of negative fermion masses
            oHsqm(0) = rHalf/sqm(0)     ! normalization for the derivative part
            oHsqm(1) = rHalf/sqm(1)     ! normalization for the derivative part

            fo%W(1)%comp(0) = im     * sqm(abs(ip))
            fo%W(2)%comp(0) = ip*nsf * sqm(abs(ip))
            fo%W(3)%comp(0) = im*nsf * sqm(abs(im))
            fo%W(4)%comp(0) = ip     * sqm(abs(im))    
            do i = 1, size(p(0))
               fo%W(1)%comp(i) = oHsqm(abs(ip))*
     &            (+im*    (p(0)%comp(i) +    p(3)%comp(i))
     &             +ip*nsf*(p(1)%comp(i) + ci*p(2)%comp(i))) 

               fo%W(2)%comp(i) = oHsqm(abs(ip))*
     &            (+im*    (p(1)%comp(i) - ci*p(2)%comp(i))
     &             +ip*nsf*(p(0)%comp(i) -    p(3)%comp(i)))

               fo%W(3)%comp(i) = oHsqm(abs(im))*
     &            (+im*nsf*(p(0)%comp(i) -    p(3)%comp(i))
     &             -ip*    (p(1)%comp(i) + ci*p(2)%comp(i))) 

               fo%W(4)%comp(i) = oHsqm(abs(im))*
     &            (-im*nsf*(p(1)%comp(i) - ci*p(2)%comp(i))
     &             +ip*    (p(0)%comp(i) +    p(3)%comp(i)))
            enddo     
         else
            ip = (3+nh)/2
            im = (3-nh)/2

            sf(1) = qreal(1+nsf+(1-nsf)*nh)*rHalf
            sf(2) = qreal(1+nsf-(1-nsf)*nh)*rHalf
            omega(1) = sqrt(p(0)+pp)
            omega(2) = fmass/omega(1)
            sfomeg(1) = sf(1)*omega(ip)
            sfomeg(2) = sf(2)*omega(im)

            pp3 = pp + p(3)
            pp3%comp(0) = max(pp3%comp(0)%re, rZero)
                      
            if ((p(1)%comp(0)%re.eq.0d0).and.(p(2)%comp(0)%re.eq.0d0)
     &          .and.(p(3)%comp(0)%re.lt.0d0)) then
               oHsqm(0) = rHalf*sfomeg(1)%comp(0)/pp%comp(0)
               oHsqm(1) = rHalf*sfomeg(2)%comp(0)/pp%comp(0)
               chi(2)%comp(0) = qcmplx(-nh)
            else
               oHsqm(0:1) = (/0E0_16,0E0_16/)
               chi(1) = sqrt(rHalf*pp3/pp)
               chi(2) = (qreal(nh)*p(1) - ci*p(2))/(sqrt(rTwo*pp*pp3))
            endif

            fo%W(1) = sfomeg(2)*chi(im)
     &            + (p(1) + nh*ci*p(2))*oHsqm(1)*chi(ip)
            fo%W(2) = sfomeg(2)*chi(ip)
     &            - (p(1) + nh*ci*p(2))*oHsqm(1)*chi(im)
            fo%W(3) = sfomeg(1)*chi(im)
     &            + (p(1) + nh*ci*p(2))*oHsqm(0)*chi(ip)
            fo%W(4) = sfomeg(1)*chi(ip)
     &            - (p(1) + nh*ci*p(2))*oHsqm(0)*chi(im)
         endif

      else

c        sqp0p3 is 0 for a momentum along -z, as in HELAS, and so are its
c        derivatives where p0 + p3 = 0 (the dual square root of 0 has none)
         call sqp0p3%initZERO()
         if((p(1)%comp(0)%re.ne.0d0).or.(p(2)%comp(0)%re.ne.0d0).or.
     &      (p(3)%comp(0)%re.gt.0d0)) then
            if (p(0)%comp(0)%re+p(3)%comp(0)%re.gt.rZero)
     &         sqp0p3 = sqrt(p(0)+p(3))*nsf
            sqp0p3%comp(0) = sqrt(max(p(0)%comp(0)%re+
     &                        p(3)%comp(0)%re,rZero))*nsf
         end if
         chi(1) = sqp0p3
         if (sqp0p3%comp(0)%re.eq.rZero) then
            chi(2) = qcmplx(-nhel)*sqrt(rTwo*p(0))
         else
            chi(2) = (nh*p(1) - ci*p(2))/sqp0p3
         endif
         if (nh.eq.1) then
            fo%W(1) = chi(1)
            fo%W(2) = chi(2)
         else
            fo%W(3) = chi(2)
            fo%W(4) = chi(1)
         endif

      endif
c
      return
      end subroutine mp_oxxxxx

      subroutine mp_vxxxxx(p,vmass,nhel,nsv,vc)
      use ALOHA_OBJECT
c
c This subroutine computes a VECTOR wavefunction.
c
c input:
c       complex p(0:3)         : four-momentum of vector boson
c       real    fmass          : mass          of vector boson
c       integer nhel = -1, 0, 1: helicity      of vector boson
c       integer nsv  = -1 or 1 : +1 for final, -1 for initial
c
c output:
c       complex vc%W(4)          : vector wavefunction       epsilon^mu(p)
c
      implicit none
      type(mp_aloha) vc
      type(MP_Dual)::p(0:3),pp,pt,pt2,pzpt,emp
      real*16 vmass,hel,hel0,sqh
      integer nhel,nsv,nsvahl
      integer i

      real*16 rZero, rHalf, rOne, rTwo
      complex*32 ci
      parameter( rZero = 0.0E0_16, rHalf = 0.5E0_16 )
      parameter( rOne = 1.0E0_16, rTwo = 2.0E0_16 )
      parameter( ci = cmplx(0.0E0_16,1.0E0_16,kind=16) )

      sqh = sqrt(rHalf)
      hel = qreal(nhel)
      nsvahl = nsv*abs(hel)
      pt2 = p(1)**2+p(2)**2
      pp = sqrt(pt2+p(3)**2)
      if ((p(1)%comp(0).ne.(0E0_16,0E0_16))
     $     .or.(p(2)%comp(0).ne.(0E0_16,0E0_16))) then
         pt = sqrt(pt2)
      else
         CALL pt%initZERO()
      endif

      pp%comp(0) = min(p(0)%comp(0)%re,pp%comp(0)%re)
      pt%comp(0) = min(pp%comp(0)%re,pt%comp(0)%re)

c     Convention for dual computations (same as loop)
      vc%P(0) = p(0)*nsv
      vc%P(1) = p(1)*nsv
      vc%P(2) = p(2)*nsv
      vc%P(3) = p(3)*nsv
      do i = 1,4
         CALL vc%W(i)%initZERO()
      enddo
      vc%flv_index = 0

      if (vmass.ne.rZero) then

         hel0 = rOne-abs(hel)

         if ( pp%comp(0)%re.eq.rZero ) then
            vc%W(1)%comp(0) = qcmplx( rZero )
            vc%W(2)%comp(0) = qcmplx(-hel*sqh )
            vc%W(3)%comp(0) = qcmplx( rZero , nsvahl*sqh )
            vc%W(4)%comp(0) = qcmplx( hel0 )

         else
            emp = p(0)/(vmass*pp)
            vc%W(1) = hel0*pp/vmass
            vc%W(4) = hel0*p(3)*emp+hel*pt/pp*sqh
            if ( pt%comp(0)%re.ne.rZero ) then
               pzpt = p(3)/(pp*pt)*sqh*hel
               vc%W(2) = hel0*p(1)*emp-p(1)*pzpt
     &                   -ci*nsvahl*p(2)/pt*sqh
               vc%W(3) = hel0*p(2)*emp-p(2)*pzpt
     &                   +ci*nsvahl*p(1)/pt*sqh
            else
               vc%W(2)%comp(0) = qcmplx( -hel*sqh )
               vc%W(3)%comp(0) = qcmplx( rZero , 
     &                   nsvahl*sign(sqh,p(3)%comp(0)%re) )
            endif

         endif

      else
         pp = p(0)
         CALL vc%W(1)%initZERO()
         vc%W(4) = hel*pt/pp*sqh
         if ( pt%comp(0)%re.ne.rZero ) then
            pzpt = p(3)/(pp*pt)*sqh*hel
            vc%W(2) = -p(1)*pzpt-ci*nsv*p(2)/pt*sqh
            vc%W(3) = -p(2)*pzpt+ci*nsv*p(1)/pt*sqh
        else
            vc%W(2)%comp(0) = qcmplx( -hel*sqh )
            vc%W(3)%comp(0) = qcmplx( rZero , 
     &                   nsv*sign(sqh,p(3)%comp(0)%re) )
         endif

      endif
c
      return
      end subroutine mp_vxxxxx

      subroutine mp_sxxxxx(p,nss,sc)
      use ALOHA_OBJECT
c
c This subroutine computes a complex SCALAR wavefunction.
c
c input:
c       complex    p(0:3)      : four-momentum of scalar boson
c       integer nss  = -1 or 1 : +1 for final, -1 for initial
c
c output:
c       complex sc%W(1)          : scalar wavefunction                   s
c
      implicit none
      type(mp_aloha) sc
      type(MP_Dual)::p(0:3)
      integer nss
      integer i

      real*16 rOne
      parameter( rOne = 1.0E0_16 )

c     Convention for dual computations (same as loop)
      sc%P(0) = p(0)*nss
      sc%P(1) = p(1)*nss
      sc%P(2) = p(2)*nss
      sc%P(3) = p(3)*nss
      CALL sc%W(1)%initZERO()
      sc%flv_index = 0

      sc%W(1)%comp(0) = qcmplx( rOne )

c
      return
      end subroutine mp_sxxxxx

