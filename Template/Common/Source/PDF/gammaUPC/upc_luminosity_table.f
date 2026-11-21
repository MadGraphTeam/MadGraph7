      program upc_luminosity_table
c***************************************************************************
c     Tabulate the gamma-UPC two-photon luminosity for madspace (mg7).
c
c     madevent multiplies the matrix element of a photon-photon subprocess
c     by photonpdfsquare(x1,x2) (auto_dsig, |lpp|=2 on both beams and a
c     gamma-UPC pdlabel), x being the fraction of the energy of the whole
c     beam (ebeam is the energy of the ion). That luminosity does not
c     factorise into one density per beam (survival factor, neutron
c     tagging), so madspace reads it as one two-dimensional table.
c
c     mg7 integrates per nucleon: e_cm = sqrt(s_NN) and xx = A x. In those
c     variables dx1 dx2 L(x1,x2) = dxx1 dxx2 L(xx1/A1,xx2/A2)/(A1 A2), so
c     the table holds
c        xx1 xx2 L(xx1/A1, xx2/A2) / (A1 A2)
c     (the x f convention of an LHAPDF grid) on a log grid of xx1 (the x
c     axis of the file) and xx2 (its q axis), particle id 22.
c
c     Input (stdin, list directed):
c        pdlabel                          gamma-UPC label, e.g. chff
c        nb_proton1 nb_neutron1 nb_proton2 nb_neutron2
c        ebeam1 ebeam2                    beam (ion) energies in GeV
c        xmin nperdecade                  xx range [xmin,1], knots/decade
c        output                           path of the grid file
c***************************************************************************
      implicit none
      include '../pdf.inc'
      integer lpp(2)
      double precision ebeamMG5(2),xbk(2),q2fact(2)
      common/to_collider/ebeamMG5,xbk,q2fact,lpp
      integer nb_proton(2), nb_neutron(2)
      common/to_heavyion_pdg/ nb_proton, nb_neutron

      double precision photonpdfsquare
      external photonpdfsquare

      character*7 label
      character*512 output
      double precision xmin, a1, a2, lum
      integer nperdecade, nknot, i, j
      double precision, allocatable :: xk(:)

      read(*,*) label
      read(*,*) nb_proton(1), nb_neutron(1), nb_proton(2), nb_neutron(2)
      read(*,*) ebeamMG5(1), ebeamMG5(2)
      read(*,*) xmin, nperdecade
      read(*,'(a)') output

      pdlabel = label
      pdsublabel(1) = label
      pdsublabel(2) = label
      lpp(1) = 2
      lpp(2) = 2
      a1 = dble(nb_proton(1) + nb_neutron(1))
      a2 = dble(nb_proton(2) + nb_neutron(2))

      if (xmin.le.0d0 .or. xmin.ge.1d0 .or. nperdecade.lt.1) then
         write(*,*) 'Error: invalid xmin/nperdecade', xmin, nperdecade
         stop 1
      endif
      nknot = max(2, nint(-dlog10(xmin)*nperdecade) + 1)
      allocate(xk(nknot))
      do i = 1, nknot
         xk(i) = xmin**(dble(nknot-i)/dble(nknot-1))
      enddo
      xk(nknot) = 1d0

      open(unit=31, file=trim(output), status='replace')
      write(31,'(a)') '# gamma-UPC two-photon luminosity, written by'
      write(31,'(a)') '# Source/PDF/gammaUPC/upc_luminosity_table.f'
      write(31,'(a,a)') '# pdlabel: ', label
      write(31,'(a,4i5)') '# nb_proton/nb_neutron: ', nb_proton(1),
     &     nb_neutron(1), nb_proton(2), nb_neutron(2)
      write(31,'(a,2es24.16)') '# ebeam: ', ebeamMG5(1), ebeamMG5(2)
      write(31,'(a)') 'PdfType: central'
      write(31,'(a)') 'Format: lhagrid1'
      write(31,'(a)') '---'
      write(31,'(*(es24.16))') (xk(i), i = 1, nknot)
      write(31,'(*(es24.16))') (xk(i), i = 1, nknot)
      write(31,'(a)') '22'
c     the q (here xx2) index runs fastest
      do i = 1, nknot
         do j = 1, nknot
            lum = photonpdfsquare(xk(i)/a1, xk(j)/a2)
            if (lum.ne.lum) then
               write(*,*) 'Error: gamma-UPC returned NaN at', xk(i), xk(j)
               stop 1
            endif
            write(31,'(es24.16)') xk(i)*xk(j)*lum/(a1*a2)
         enddo
      enddo
      write(31,'(a)') '---'
      close(31)
      end
