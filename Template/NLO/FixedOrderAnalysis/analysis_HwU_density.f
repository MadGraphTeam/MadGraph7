c
c The default fixed-order analysis of an output made with --density=i,j
c (docs/nlo_density_usage_plan.md). With density_weights = True in the
c run_card, each weight comes with the elements of the spin-density
c matrix of the density legs (the weights labelled 'dens ...'), so that
c each histogram below also gives, per bin, the density matrix averaged
c over the bin. After the run,
c    python bin/internal/histograms.py Events/run_XX/MADatNLO.HwU --density
c turns them into the normalised density matrix, its elements and the
c derived quantities (f_L, f_0, f_R for one vector boson; polarisations
c and spin correlations for two spin-1/2 particles) of each bin.
c
c Histograms: the total rate; the transverse momentum and the rapidity
c of each density leg (the first two); for two density legs, their pair
c mass, transverse momentum and azimuthal separation, and the map of
c (theta, beta) of the first one in the rest frame of the pair (reached
c by a boost from the lab frame): theta its angle with the beam (+z),
c beta its velocity, one histogram of theta (18 bins, 0 to 180 degrees)
c per bin of beta (10 bins, 0 to 1), and the same map in (theta, pair
c mass): theta in 20 bins, the mass in 24 bins from 346 to 1000 GeV
c (finer near the threshold of a t t~ pair); and the transverse
c momentum of the hardest anti-kT jet (R=0.4, pT > 20 GeV, |eta| < 4.5),
c with all the QCD partons. The density legs are found by their PDG
c codes (density_pdg.inc of the P directory).
c
c This uses the HwU package and generates histograms in the HwU/GnuPlot
c format. After running, the histograms can be found in the
c Events/run_XX/ directory.
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_begin(nwgt,weights_info)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      include 'density_pdg.inc'
      integer nwgt
      character*(*) weights_info(*)
      character*20 name(2)
      character*50 title
      integer k,n,nbeta,nmass
      parameter (nbeta=10,nmass=24)
      double precision medge(0:nmass)
      data medge /346d0,348d0,351d0,354d0,358d0,362d0,367d0,372d0,378d0
     $     ,385d0,393d0,402d0,412d0,425d0,440d0,460d0,485d0,515d0,550d0
     $     ,600d0,660d0,730d0,810d0,900d0,1000d0/
      call HwU_inithist(nwgt,weights_info)
      call HwU_book(1,'total rate',1,0.5d0,1.5d0)
      n=min(2,n_dens_legs)
      do k=1,n
         write (name(k),'(a,i0)') 'leg ',dens_leg(k)
         call HwU_book(2*k,'pt '//trim(name(k)),25,0d0,500d0)
         call HwU_book(2*k+1,'y '//trim(name(k)),20,-5d0,5d0)
      enddo
      if (n.eq.2) then
         call HwU_book(6,'m '//trim(name(1))//' '//trim(name(2)),30,0d0
     $        ,1500d0)
         call HwU_book(7,'pt '//trim(name(1))//' '//trim(name(2)),25
     $        ,0d0,500d0)
         call HwU_book(8,'dphi '//trim(name(1))//' '//trim(name(2)),20
     $        ,0d0,3.1415926535897932d0)
      endif
      call HwU_book(9,'pt j1',25,0d0,500d0)
      if (n.eq.2) then
         do k=1,nbeta
            write (title,'(a,a,a,a,a,f3.1,a,f3.1)') 'theta ',
     $           trim(name(1)),' ',trim(name(2)),' beta ',dble(k-1)
     $           /dble(nbeta),'-',dble(k)/dble(nbeta)
            call HwU_book(9+k,trim(title),18,0d0,180d0)
         enddo
         do k=1,nmass
            write (title,'(a,a,a,a,a,i0,a,i0)') 'theta ',
     $           trim(name(1)),' ',trim(name(2)),' mass ',nint(medge(k
     $           -1)),'-',nint(medge(k))
            call HwU_book(9+nbeta+k,trim(title),20,0d0,180d0)
         enddo
      endif
      return
      end


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_end(dummy)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      double precision dummy
      call HwU_write_file
      return
      end


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_fill(p,istatus,ipdg,wgts,ibody)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c The momenta p (lab frame: E, px, py, pz, mass) of the (n+1)-body
c particles (for n-body contributions one of them is zero), their
c istatus (-1 initial, 1 final state) and PDG codes, the weights (the
c central one, the scale and PDF ones and the density-matrix ones) and
c ibody (1: (n+1)-body, 2: n-body without the Born, 3: Born).
      implicit none
      include 'nexternal.inc'
      include 'density_pdg.inc'
      integer istatus(nexternal),iPDG(nexternal),ibody
      double precision p(0:4,nexternal),wgts(*)
      integer i,j,k,n,ileg(2),nQCD,njet,jet(nexternal),nbeta,ibeta
     $     ,nmass
      parameter (nbeta=10,nmass=24)
      double precision medge(0:nmass),mpair
      data medge /346d0,348d0,351d0,354d0,358d0,362d0,367d0,372d0,378d0
     $     ,385d0,393d0,402d0,412d0,425d0,440d0,460d0,485d0,515d0,550d0
     $     ,600d0,660d0,730d0,810d0,900d0,1000d0/
      logical used(nexternal)
      double precision pQCD(0:3,nexternal),pjet(0:3,nexternal),ppair(0:3)
     $     ,dphi,pi,pboost(0:3),pstar(0:3),beta,theta,pmod
      parameter (pi=3.1415926535897932d0)
      double precision dens_pt,dens_rap,dens_mass
      external dens_pt,dens_rap,dens_mass
      call HwU_fill(1,1d0,wgts)
c the density legs: the first final-state particles with their PDG codes
      n=min(2,n_dens_legs)
      do i=1,nexternal
         used(i)=.false.
      enddo
      do k=1,n
         ileg(k)=0
         do i=1,nexternal
            if (istatus(i).ne.1 .or. used(i)) cycle
            if (ipdg(i).eq.dens_pdg(k) .and. p(0,i).gt.0d0) then
               ileg(k)=i
               used(i)=.true.
               exit
            endif
         enddo
         if (ileg(k).eq.0) then
            write (*,*) 'ERROR in analysis_HwU_density: no final-state'/
     $           /' particle with the PDG code of density leg',dens_leg(k)
     $           ,dens_pdg(k)
            stop 1
         endif
         call HwU_fill(2*k,dens_pt(p(0,ileg(k))),wgts)
         call HwU_fill(2*k+1,dens_rap(p(0,ileg(k))),wgts)
      enddo
      if (n.eq.2) then
         do j=0,3
            ppair(j)=p(j,ileg(1))+p(j,ileg(2))
         enddo
         call HwU_fill(6,dens_mass(ppair),wgts)
         call HwU_fill(7,dens_pt(ppair),wgts)
         dphi=abs(atan2(p(2,ileg(1)),p(1,ileg(1)))
     $        -atan2(p(2,ileg(2)),p(1,ileg(2))))
         if (dphi.gt.pi) dphi=2d0*pi-dphi
         call HwU_fill(8,dphi,wgts)
c the first density leg in the rest frame of the pair
         pboost(0)=ppair(0)
         do j=1,3
            pboost(j)=-ppair(j)
         enddo
         call boostx(p(0,ileg(1)),pboost,pstar)
         pmod=sqrt(pstar(1)**2+pstar(2)**2+pstar(3)**2)
         beta=pmod/pstar(0)
         theta=acos(max(-1d0,min(1d0,pstar(3)/pmod)))*180d0/pi
         ibeta=min(int(beta*nbeta)+1,nbeta)
         call HwU_fill(9+ibeta,theta,wgts)
         mpair=dens_mass(ppair)
         do k=1,nmass
            if (mpair.ge.medge(k-1) .and. mpair.lt.medge(k)) then
               call HwU_fill(9+nbeta+k,theta,wgts)
               exit
            endif
         enddo
      endif
c the hardest anti-kT jet of the QCD partons
      nQCD=0
      do i=nincoming+1,nexternal
         if (abs(ipdg(i)).le.5 .or. ipdg(i).eq.21) then
            if (p(0,i).le.0d0) cycle
            nQCD=nQCD+1
            do j=0,3
               pQCD(j,nQCD)=p(j,i)
            enddo
         endif
      enddo
      njet=0
      if (nQCD.gt.0) call amcatnlo_fastjetppgenkt_etamax(pQCD,nQCD,0.4d0
     $     ,20d0,4.5d0,-1d0,pjet,njet,jet)
      if (njet.gt.0) call HwU_fill(9,dens_pt(pjet(0,1)),wgts)
      return
      end


      double precision function dens_pt(p)
      implicit none
      double precision p(0:3)
      dens_pt=sqrt(p(1)**2+p(2)**2)
      end


      double precision function dens_rap(p)
      implicit none
      double precision p(0:3)
      if (p(0)-p(3).le.0d0 .or. p(0)+p(3).le.0d0) then
         dens_rap=sign(99d0,p(3))
      else
         dens_rap=0.5d0*log((p(0)+p(3))/(p(0)-p(3)))
      endif
      end


      double precision function dens_mass(p)
      implicit none
      double precision p(0:3)
      dens_mass=sqrt(max(p(0)**2-p(1)**2-p(2)**2-p(3)**2,0d0))
      end
