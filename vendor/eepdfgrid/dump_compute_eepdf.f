c     Dump the full dressed-lepton density -- the reduced grid times its
c     analytic (1-x) factor, i.e. what compute_eepdf() in
c     Template/LO/Source/PDF/pdg2pdf.f returns -- on the (x, Q2) points of a
c     file. Reference for madspace's EEPartonDensity.
c
c     compute_eepdf is not called directly because it sits in pdg2pdf.f with
c     the whole PDF machinery behind it; the two lines it adds to
c     eepdf_tilde() are reproduced here instead, ps_expo included.
c
c     gfortran -w -ffixed-line-length-132 -o dump_compute dump_compute_eepdf.f \
c         <set>/eepdf.f <set>/gridpdfaux.f \
c         Template/Common/Source/PDF/dfint.f Template/Common/Source/PDF/kerset.f abend.f
      program dump_compute
      implicit none
      integer maxpt
      parameter (maxpt=200000)
      double precision x(maxpt), q2(maxpt)
      double precision eepdf_tilde, eepdf_tilde_power
      external eepdf_tilde, eepdf_tilde_power
      double precision val(16), omx, kexp, ps_expo
c     get_ee_expo() of pdg2pdf.f
      parameter (ps_expo=0.96d0)
      integer npt, i, k, n, ia, ib, ipart, ibeam
      character*256 fname

      call getarg(1, fname)
      open(unit=11, file=fname, status='old')
      read(11,*) npt
      do i = 1, npt
         read(11,*) x(i), q2(i)
      enddo
      close(11)

      do i = 1, npt
         omx = 1d0 - x(i)
         k = 0
         do n = 1, 4
            do ia = 1, 2
               do ib = 1, 2
                  ipart = -11
                  if (ia.eq.2) ipart = 11
                  ibeam = -11
                  if (ib.eq.2) ibeam = 11
                  k = k + 1
                  kexp = eepdf_tilde_power(q2(i), n, ipart, ibeam)
                  val(k) = eepdf_tilde(x(i), q2(i), n, ipart, ibeam)
     &                 * omx**(-kexp + ps_expo)
               enddo
            enddo
         enddo
         write(*,'(18(1x,1p,e24.16))') x(i), q2(i), (val(k), k=1,16)
      enddo
      end
