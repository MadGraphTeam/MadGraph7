      double precision function photonpdfsquare(x1,x2)
c***************************************************************************
c     Based on pdf.f, wrapper for calling the pdf of MCFM
c     ih is now signed <0 for antiparticles
c     if ih<0 does not have a dedicated pdf, then the one for ih>0 will be called
c     and the sign of ipdg flipped accordingly.
c
c     ibeam is the beam identity 1/2
c      if set to -1/-2 it meand that ipdg should not be flipped even if ih<0
c      useful for re-weighting
c***************************************************************************
      USE ElasticPhotonPhotonFlux
      implicit none
c
c     Arguments
c
      DOUBLE  PRECISION x1,x2
C
C     Include
C
C      include '../pdf.inc'
C     Common block
      include '../pdf.inc'
c      character*7 pdlabel,epa_label
c      character*7 pdsublabel(2)
c      integer lhaid
c      common/to_pdf/lhaid,pdlabel,epa_label,pdsublabel

      double precision xx1,xx2

      integer nb_proton(2), nb_neutron(2) 
      common/to_heavyion_pdg/ nb_proton, nb_neutron
      integer nb_hadron(2)
C      

      integer i,j
      double precision xlast(2,2),pdflast(2)
      character*7 pdlabellast(2)
      integer ireuse
      save xlast,pdflast,pdlabellast
      data xlast/4*-99d9/
      data pdflast/2*-99d9/
      data pdlabellast/2*'abcdefg'/

c     collider configuration
      integer lpp(2)
      double precision ebeamMG5(2),xbk(2),q2fact(2)
      common/to_collider/ebeamMG5,xbk,q2fact,lpp

      do i=1,2
         nb_hadron(i) = nb_proton(i)+nb_neutron(i)
      enddo
      xx1=x1*nb_hadron(1)
      xx2=x2*nb_hadron(2)
c     Make sure we have a reasonable Bjorken x. Note that even though
c     x=0 is not reasonable, we prefer to simply return photonpdfsquare=0
c     instead of stopping the code, as this might accidentally happen.
      if (xx1.eq.0d0.or.xx2.eq.0d0) then
         photonpdfsquare=0d0
         return
      elseif (xx1.lt.0d0 .or. xx1.gt.1d0) then
         if (nb_hadron(1).eq.1.or.x1.lt.0d0) then
            write (*,*) 'PDF#1 not supported for Bjorken x ', xx1
            open(unit=26,file='../../../error',status='unknown')
            write(26,*) 'Error: PDF#1 not supported for Bjorken x ',xx1
            stop 1
         else
            photonpdfsquare = 0d0
            return
         endif
      elseif (xx2.lt.0d0 .or. xx2.gt.1d0) then
         if (nb_hadron(2).eq.1.or.x2.lt.0d0) then
            write (*,*) 'PDF#2 not supported for Bjorken x ', xx2
            open(unit=26,file='../../../error',status='unknown')
            write(26,*) 'Error: PDF#2 not supported for Bjorken x ',xx2
            stop 1
         else
            photonpdfsquare = 0d0
            return
         endif
      endif

      ireuse = 0
      do i=1,2
c     Check if result can be reused since any of last two calls
         if (xx1.eq.xlast(1,i) .and. xx2.eq.xlast(2,i) .and.
     $        pdlabel.eq.pdlabellast(i)) then
            ireuse = i
         endif
      enddo

c     Reuse previous result, if possible
      if (ireuse.gt.0)then
         if (pdflast(ireuse).ne.-99d9) then
            photonpdfsquare = pdflast(ireuse)
            return 
         endif
      endif

c     Bjorken x and/or PDF set are not
c     identical to the saved values: this means a new event and we
c     should reset everything to compute new PDF values. Also, determine
c     if we should fill ireuse=1 or ireuse=2.
      if (ireuse.eq.0.and.xlast(1,1).ne.-99d9.and.xlast(2,1).ne.-99d9
     $     .and.xlast(1,2).ne.-99d9.and.xlast(2,2).ne.-99d9)then
         do i=1,2
            xlast(1:2,i)=-99d9
            pdflast(i)=-99d9
            pdlabellast(i)='abcdefg'
         enddo
c     everything has been reset. Now set ireuse=1 to fill the first
c     arrays of saved values below
         ireuse=1
      else if(ireuse.eq.0.and.xlast(1,1).ne.-99d9
     $        .and.xlast(2,1).ne.-99d9)then
c     This is first call after everything has been reset, so the first
c     arrays are already filled with the saved values (hence
c     xlast(1,1).ne.-99d9 and xlast(2,1).ne.-99d9). 
c     Fill the second arrays of saved values (done
c     below) by setting ireuse=2
         ireuse=2
      else if(ireuse.eq.0)then
c     Special: only used for the very first call to this function:
c     xlast(1,i), xlast(2,i) are initialized as data statements to be equal to -99d9
         ireuse=1
      endif

c     Give the current values to the arrays that should be
c     saved. 'pdflast' is filled below.
      xlast(1,ireuse)=xx1
      xlast(2,ireuse)=xx2
      pdlabellast(ireuse)=pdlabel

      neutron_tagging(1:2)=-2
      if(pdlabel(1:3).eq.'edf') then
         USE_CHARGEFORMFACTOR4PHOTON=.FALSE.
         if(pdlabel(4:4).ne.'f')then
c     forward neutron tagging
            if(pdlabel(5:5).ne.'n'.or.pdlabel(7:7).ne.'n')then
               WRITE(*,*)"Error: do not know pdlabel = ",pdlabel
               STOP 3
            endif
            if(pdlabel(4:4).eq.'i')then
c 0n+Xn
               neutron_tagging(1)=-2
            elseif(pdlabel(4:4).eq.'x')then
c Xn
               neutron_tagging(1)=-1
            elseif(pdlabel(4:4).eq.'0')then
c 0n
               neutron_tagging(1)=0
            elseif(pdlabel(4:4).eq.'1')then
c 1n
               neutron_tagging(1)=1
            elseif(pdlabel(4:4).eq.'2')then
c 2n
               neutron_tagging(1)=2
            elseif(pdlabel(4:4).eq.'3')then
c 3n
               neutron_tagging(1)=3
            elseif(pdlabel(4:4).eq.'4')then
c 4n
               neutron_tagging(1)=4
            else
               WRITE(*,*)"Error: do not know pdlabel = ",pdlabel
               STOP 4
            endif
            if(pdlabel(6:6).eq.'i')then
c 0n+Xn
               neutron_tagging(2)=-2
            elseif(pdlabel(6:6).eq.'x')then
c Xn
               neutron_tagging(2)=-1
            elseif(pdlabel(6:6).eq.'0')then
c 0n
               neutron_tagging(2)=0
            elseif(pdlabel(6:6).eq.'1')then
c 1n
               neutron_tagging(2)=1
            elseif(pdlabel(6:6).eq.'2')then
c 2n
               neutron_tagging(2)=2
            elseif(pdlabel(6:6).eq.'3')then
c 3n
               neutron_tagging(2)=3
            elseif(pdlabel(6:6).eq.'4')then
c 4n
               neutron_tagging(2)=4
            else
               WRITE(*,*)"Error: do not know pdlabel = ",pdlabel
               STOP 5
            endif
         else
            if(pdlabel(5:7).ne.'   ')then
               WRITE(*,*)"Error: do not know pdlabel = ",pdlabel
               STOP 9
            endif
         endif
      elseif(pdlabel(1:3).eq.'chf') then
         USE_CHARGEFORMFACTOR4PHOTON=.TRUE.
         if(pdlabel(4:4).ne.'f')then
c     forward neutron tagging
            if(pdlabel(5:5).ne.'n'.or.pdlabel(7:7).ne.'n')then
               WRITE(*,*)"Error: do not know pdlabel = ",pdlabel
               STOP 6
            endif
            if(pdlabel(4:4).eq.'i')then
c 0n+Xn
               neutron_tagging(1)=-2
            elseif(pdlabel(4:4).eq.'x')then
c Xn
               neutron_tagging(1)=-1
            elseif(pdlabel(4:4).eq.'0')then
c 0n
               neutron_tagging(1)=0
            elseif(pdlabel(4:4).eq.'1')then
c 1n
               neutron_tagging(1)=1
            elseif(pdlabel(4:4).eq.'2')then
c 2n
               neutron_tagging(1)=2
            elseif(pdlabel(4:4).eq.'3')then
c 3n
               neutron_tagging(1)=3
            elseif(pdlabel(4:4).eq.'4')then
c 4n
               neutron_tagging(1)=4
            else
               WRITE(*,*)"Error: do not know pdlabel = ",pdlabel
               STOP 7
            endif
            if(pdlabel(6:6).eq.'i')then
c 0n+Xn
               neutron_tagging(2)=-2
            elseif(pdlabel(6:6).eq.'x')then
c Xn
               neutron_tagging(2)=-1
            elseif(pdlabel(6:6).eq.'0')then
c 0n
               neutron_tagging(2)=0
            elseif(pdlabel(6:6).eq.'1')then
c 1n
               neutron_tagging(2)=1
            elseif(pdlabel(6:6).eq.'2')then
c 2n
               neutron_tagging(2)=2
            elseif(pdlabel(6:6).eq.'3')then
c 3n
               neutron_tagging(2)=3
            elseif(pdlabel(6:6).eq.'4')then
c 4n
               neutron_tagging(2)=4
            else
               WRITE(*,*)"Error: do not know pdlabel = ",pdlabel
               STOP 8
            endif
         else
            if(pdlabel(5:7).ne.'   ')then
               WRITE(*,*)"Error: do not know pdlabel = ",pdlabel
               STOP 10
            endif
         endif
      else
         WRITE(*,*)"Error: do not know pdlabel = ",pdlabel
         STOP 2
      endif

c     write(*,*) 'running gamma-UPC'

      IF(nb_hadron(1).eq.1.and.nb_hadron(2).eq.1)THEN
         pdflast(ireuse)=PhotonPhotonFlux_pp(xx1,xx2)
      ELSEIF((nb_hadron(1).eq.1.and.nb_hadron(2).gt.1).or.
     $        (nb_hadron(2).eq.1.and.nb_hadron(1).gt.1))THEN
         pdflast(ireuse)=PhotonPhotonFlux_pA_WoodsSaxon(xx1,xx2)
      ELSEIF(nb_hadron(1).gt.1.and.nb_hadron(2).gt.1)THEN
         pdflast(ireuse)=PhotonPhotonFlux_AB_WoodsSaxon(xx1,xx2)
      ELSE
         WRITE(*,*)"Error: do not know nb_hadron(1:2) = ",nb_hadron(1:2)
         STOP 3
      ENDIF
      ! the particular normalisation for MG5 in heavy ion mode
      pdflast(ireuse)=pdflast(ireuse)*nb_hadron(1)*nb_hadron(2)
      photonpdfsquare=pdflast(ireuse)
      
      return
      end
      

      subroutine Get_nucleus_RA(nb_p,nb_n,RAI)
      USE ElasticPhotonPhotonFlux
      implicit none
      ! number of protons and neutrons in the nucleus
      integer nb_p,nb_n
      ! radius of the nucleus in unit of GeV-1
      double precision RAI
      integer nb_pn
      double precision RRR, AAA, WWW, Aval, Zval
      character*7 nucleus_name
      ! the charge radius of proton (in fm)
      double precision Rproto
      parameter (Rproto=0.877d0)
      ! convert from GeV-1 to fm
      double precision GeVm12fm1
      parameter (GeVm12fm1=0.1973d0)
      if(nb_p.eq.1.and.nb_n.eq.0)then
         RAI=Rproto/GeVm12fm1
         return
      endif
      nb_pn=nb_p+nb_n
      nucleus_name=GetASymbol(nb_pn,nb_p)
      CALL GetNuclearInfo(nucleus_name,Aval,Zval,RRR,AAA,WWW)
      RAI=RRR/GeVm12fm1
      return
      end

      double precision function gammaupc_single_flux(x, ebeam, ichff)
c     The single-photon number density dN/dx of an elastic proton of energy
c     ebeam (gamma-UPC, without the survival factor of hadronic overlap):
c     EDFF (ichff = 0) or ChFF (ichff = 1). Used by GPD processes, where the
c     other proton is probed by its GPDs (Source/PDF/pdg2pdf.f, lpp = 5).
      use ElasticPhotonPhotonFlux
      implicit none
      double precision x, ebeam, gamma
      integer ichff
      double precision mproton
      parameter (mproton=0.938272081d0)
      gamma = ebeam/mproton
      if (ichff.eq.1) then
         gammaupc_single_flux = PhotonFlux_proton_ChFF_nob(x, gamma)
      else
         gammaupc_single_flux = PhotonFlux_proton_EDFF_nob(x, gamma)
      endif
      return
      end


      double precision function gammaupc_single_flux_pp(x, ebeam,
     &     ichff, sqrts)
c     The single-photon number density dN/dx of an elastic proton of
c     energy ebeam that leaves the other proton of a pp collision at
c     sqrts intact: the photon density N(E,b) at distance b from its
c     emitter times the probability of no inelastic pp interaction of
c     arXiv:2207.03012, eq. (10) (as in PhotonPhotonFlux_pp),
c       P(b) = |1 - Gamma(s,b)|^2,  Gamma = exp(-b^2/(2 b0)),
c       b0 = 9.81 + 0.211 ln s + 0.0185 ln^2 s  GeV^-2 (s in GeV^2),
c     with b the pp impact parameter: the photon is absorbed at the
c     centre of the other proton (GPD processes, lpp = 5; the size of
c     the gamma p interaction region changes it by less than 1%).
c     EDFF (ichff = 0, b > R_p) or ChFF (ichff = 1, dipole, b > 0).
c     Computed as the flux without survival, gammaupc_single_flux,
c     minus alpha/x int d^2b N(E,b) (2 Gamma - Gamma^2).
      use ElasticPhotonPhotonFlux
      implicit none
      double precision x, ebeam, sqrts
      integer ichff
      double precision mproton, gevm1fm, rproton, pi, hpan
      parameter (mproton=0.938272081d0, gevm1fm=0.1973d0)
      parameter (rproton=0.877d0, pi=3.14159265358979323846d0)
      parameter (hpan=0.5d0)
      integer ngl
      parameter (ngl=10)
      double precision xg(ngl), wg(ngl)
      logical init
      data init/.true./
      save init, xg, wg
      integer npan, ipan, k
      double precision gamma, ega, alpha, lns, b0, ulo, uhi, h, u, b
      double precision gam, dens, deficit
      double precision gammaupc_single_flux
      external gammaupc_single_flux
      if (init) then
         call gammaupc_gauleg(ngl, xg, wg)
         init = .false.
      endif
      gammaupc_single_flux_pp = gammaupc_single_flux(x, ebeam, ichff)
      if (x.le.0d0 .or. x.ge.1d0 .or. sqrts.le.0d0) return
      gamma = ebeam/mproton
      ega = x*ebeam
      if (alphaem_elasticphoton.lt.0d0) then
         alpha = 0.0072992701d0
      else
         alpha = alphaem_elasticphoton
      endif
      lns = dlog(sqrts**2)
      b0 = 9.81d0 + 0.211d0*lns + 0.0185d0*lns**2
c     Gamma < 1e-17 beyond b^2 = 80 b0
      uhi = 0.5d0*dlog(80d0*b0)
      if (ichff.eq.1) then
         ulo = dlog(1d-6)
      else
         ulo = dlog(rproton/gevm1fm)
      endif
      if (ulo.ge.uhi) return
      npan = max(1, int((uhi - ulo)/hpan) + 1)
      h = (uhi - ulo)/npan
      deficit = 0d0
      do ipan = 1, npan
         do k = 1, ngl
            u = ulo + h*(ipan - 0.5d0 + 0.5d0*xg(k))
            b = dexp(u)
            gam = dexp(-b*b/(2d0*b0))
            if (ichff.eq.1) then
               dens = PhotonNumberDensity_ChargeFormFactor_proton(b,
     &              ega, gamma)
            else
               dens = PhotonNumberDensity(b, ega, gamma)
            endif
            deficit = deficit + 0.5d0*h*wg(k)*2d0*pi*b*b*dens
     &           *(2d0*gam - gam*gam)
         enddo
      enddo
      gammaupc_single_flux_pp = max(0d0,
     &     gammaupc_single_flux_pp - alpha/x*deficit)
      return
      end


      subroutine gammaupc_gauleg(n, xg, wg)
c     Gauss-Legendre nodes and weights on [-1, 1]
      implicit none
      integer n
      double precision xg(n), wg(n)
      integer i, j, m
      double precision z, z1, p1, p2, p3, pp, pi
      parameter (pi=3.14159265358979323846d0)
      m = (n + 1)/2
      do i = 1, m
         z = dcos(pi*(i - 0.25d0)/(n + 0.5d0))
 10      continue
         p1 = 1d0
         p2 = 0d0
         do j = 1, n
            p3 = p2
            p2 = p1
            p1 = ((2d0*j - 1d0)*z*p2 - (j - 1d0)*p3)/j
         enddo
         pp = n*(z*p1 - p2)/(z*z - 1d0)
         z1 = z
         z = z1 - p1/pp
         if (dabs(z - z1).gt.1d-15) goto 10
         xg(i) = -z
         xg(n + 1 - i) = z
         wg(i) = 2d0/((1d0 - z*z)*pp*pp)
         wg(n + 1 - i) = wg(i)
      enddo
      return
      end
