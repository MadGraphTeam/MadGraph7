c/* ******************************************************** */
c/*  Equivalent photon approximation structure function.   * */
c/*   V.M.Budnev, et al., Phys.Rep. 15C (1975) 181         * */
c/*   https://inspirehep.net/literature/95445              * */
c/*   For photon from proton (elastic limit, no DGLAP evo) * */
c/*                                                        * */
c/*  Improved Weizsaecker-Williams formula                 * */
c/*   S. Frixione, et al., Phys.Lett.B 319 (1993) 339-345  * */
c/*   http://inspirehep.net/record/359425                  * */
c/*   For photon from electron (WW approx) + higher twist  * */
c/* ******************************************************** */
c   provided by Tomasz Pierzchala - UCL

      real*8 function epa_lepton(x,q2max, mode)
      implicit none
      integer i, mode, imode
c     mode is +3/-3  for electron and +4/-4 for muon      
      real*8 x,phi_f
      real*8 xin(3:4)
      real*8 alpha
      real*8 f, q2min,q2max
      real*8 PI
      data PI/3.14159265358979323846/

      data xin/0.511d-3, 0.105658d0/ !electron mass in GeV

      alpha = .0072992701
      imode = abs(mode)
      
C     // x = omega/E = (E-E')/E
      if (x.lt.1) then
         q2min= xin(imode)*xin(imode)*x*x/(1-x)
         if(q2min.lt.q2max) then 
             f = alpha/2d0/PI*
     &           (2d0*xin(imode)*xin(imode)*x*(-1/q2min+1/q2max)+
     &           (2-2d0*x+x*x)/x*dlog(q2max/q2min))
            
         else
           f = 0. 
         endif
      else
         f= 0.
      endif
c      write (*,*) x,dsqrt(q2min),dsqrt(q2max),f
      if (f .lt. 0) f = 0
      epa_lepton = f

      end

      real*8 function epa_proton(x,q2max,beamid)
      integer i
      integer beamid
      real*8 x,phi_f
      real*8 xin
      real*8 alpha,qz
      real*8 f, qmi,qma, q2max
      real*8 PI
    
      integer nb_proton(2), nb_neutron(2)
      common/to_heavyion_pdg/ nb_proton, nb_neutron
      double precision mass_ion(2)
      common/to_heavyion_mass/mass_ion

      data PI/3.14159265358979323846/

      data xin/0.938/ ! proton mass in GeV

      alpha = .0072992701
      qz = 0.71

      if (nb_proton(beamid).ne.1.or.nb_neutron(beamid).ne.0)then
         xin = mass_ion(beamid)
         alpha = alpha * nb_proton(beamid)
      endif
    
C     // x = omega/E = (E-E')/E
      if (x.lt.1) then
         qmi= xin*xin*x*x/(1-x)
         if(qmi.lt.q2max) then          
            f = alpha/PI*(phi_f(x,q2max/qz)-phi_f(x,qmi/qz))*(1-x)/x
         else
            f=0.
         endif
      else
         f= 0.
      endif
      if (f .lt. 0) f = 0
      epa_proton= f
      end

      real*8 function phi_f(x,qq)
      real*8 x, qq
      real*8 y,qq1,f
      real*8 a,b,c

       a = 7.16
       b = -3.96
       c = .028

      qq1=1+qq
      y= x*x/(1-x)
      f=(1+a*y)*(-log(qq1/qq)+1/qq1+1/(2*qq1*qq1)+1/(3*qq1*qq1*qq1))
      f=f + (1-b)*y/(4*qq*qq1*qq1*qq1);
      f=f+ c*(1+y/4)*(log((qq1-b)/qq1)+b/qq1+b*b/(2*qq1*qq1)+
     $b*b*b/(3*qq1*qq1*qq1))
      phi_f= f
      end
  
      double precision function get_ion_pdf(pdf, pdg, nb_proton, nb_neutron)
C***********************************************************************
C     computing (heavy) ion contribution from proton PDF
C***********************************************************************      
      double precision pdf(-7:7)
      double precision tmppdf(-2:2)
      integer pdg
      integer nb_proton
      integer nb_neutron
      double precision tmp1, tmp2
 
      if (nb_proton.eq.1.and.nb_neutron.eq.0)then
         get_ion_pdf = pdf(pdg)
         return
      endif
      
      if (pdg.eq.1.or.pdg.eq.2) then
         tmp1 = pdf(1)
         tmp2 = pdf(2)
         tmppdf(1) = nb_proton * tmp1 + nb_neutron * tmp2
         tmppdf(2) = nb_proton * tmp2 + nb_neutron * tmp1
         get_ion_pdf = tmppdf(pdg)
      else if (pdg.eq.-1.or.pdg.eq.-2) then
         tmp1 = pdf(-1)
         tmp2 = pdf(-2)
         tmppdf(-1) = nb_proton * tmp1 + nb_neutron * tmp2
         tmppdf(-2) = nb_proton * tmp2 + nb_neutron * tmp1
         get_ion_pdf = tmppdf(pdg)
      else 
         get_ion_pdf = pdf(pdg)*(nb_proton+nb_neutron)
      endif
      
C     set correct PDF normalisation
      get_ion_pdf = get_ion_pdf * (nb_proton+nb_neutron)
      return
      end



      double precision function gpd_photon_flux(x, xmu, beamid)
c     The photon of a GPD process (lpp = 5): the single-photon flux of an
c     elastic proton, chosen by 'gpd_flux' of Cards/gpd_card.dat: edff
c     (gamma-UPC EDFF, default), chff (gamma-UPC ChFF) or iww (EPA). The
c     pdlabel of an lpp = 5 beam names the proton PDFs of its GPDs.
c     'gpd_gap' = pp (default) folds the edff/chff photon density in b
c     with the probability that the two protons do not interact
c     inelastically, eq. (10) of arXiv:2207.03012 (gamma-UPC); none
c     keeps the flux without it.
      implicit none
      double precision x, xmu
      integer beamid
      integer lpp(2)
      double precision ebeam(2),xbk(2),q2fact(2)
      common/to_collider/ebeam,xbk,q2fact,lpp
      double precision gammaupc_single_flux, epa_proton
      double precision gammaupc_single_flux_pp, sqrts
      external gammaupc_single_flux, epa_proton
      external gammaupc_single_flux_pp
      integer iflux, igap, ios, ieq, k
      logical init, ex
      character*256 line, fname(3)
      data init/.true./, iflux/0/, igap/1/
      data fname/'../../../Cards/gpd_card.dat', '../../Cards/gpd_card.dat',
     &     'gpd_card.dat'/
      save init, iflux, igap
      if (init) then
         init = .false.
         do k = 1, 3
            inquire(file=fname(k), exist=ex)
            if (ex) exit
         enddo
         if (ex) then
            open(88, file=fname(k), status='old')
            do
               read(88, '(a)', iostat=ios) line
               if (ios.ne.0) exit
               if (index(line,'#').gt.0) line = line(1:index(line,'#')-1)
               if (index(line,'!').gt.0) line = line(1:index(line,'!')-1)
               ieq = index(line, '=')
               if (ieq.eq.0) cycle
               if (adjustl(line(ieq+1:)).eq.'gpd_gap') then
                  if (index(line(1:ieq), 'none').gt.0) igap = 0
                  cycle
               endif
               if (adjustl(line(ieq+1:)).ne.'gpd_flux') cycle
               if (index(line(1:ieq), 'chff').gt.0) iflux = 1
               if (index(line(1:ieq), 'iww').gt.0) iflux = 2
            enddo
            close(88)
         endif
         if (iflux.eq.2 .and. igap.eq.1) then
            write(*,*) 'GPD: no gap survival with the iww photon flux'
            igap = 0
         endif
         if (igap.eq.1) then
            write(*,*) 'GPD: photon flux with the pp gap survival of',
     &           ' arXiv:2207.03012, eq. (10)'
         else
            write(*,*) 'GPD: photon flux without gap survival'
         endif
      endif
      sqrts = 2d0*dsqrt(ebeam(1)*ebeam(2))
      if (iflux.eq.2) then
         gpd_photon_flux = epa_proton(x, xmu*xmu, beamid)
      elseif (igap.eq.1) then
         gpd_photon_flux = gammaupc_single_flux_pp(x, ebeam(beamid),
     &        iflux, sqrts)
      else
         gpd_photon_flux = gammaupc_single_flux(x, ebeam(beamid), iflux)
      endif
      end
