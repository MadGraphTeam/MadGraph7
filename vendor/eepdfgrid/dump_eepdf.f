c     Dump eepdf_tilde() on the (x,Q2) points of a file, for the accuracy
c     gate of gen_eepdf_lhagrid.py: the same table is then interpolated by
c     madspace's log-bicubic kernel and the two are compared.
c
c     gfortran -w -ffixed-line-length-132 -o dump_eepdf dump_eepdf.f \
c         <set>/eepdf.f <set>/gridpdfaux.f \
c         Template/Common/Source/PDF/dfint.f Template/Common/Source/PDF/kerset.f
c
c     ./dump_eepdf points.txt > values.txt
      program dump_eepdf
      implicit none
      integer maxpt
      parameter (maxpt=200000)
      double precision x(maxpt), q2(maxpt)
      double precision eepdf_tilde
      external eepdf_tilde
      double precision val(16)
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
c        same ordering as the pid list of gen_eepdf_lhagrid.py:
c        (n, a, b) sorted, a/b = 1 -> -11, 2 -> +11
         k = 0
         do n = 1, 4
            do ia = 1, 2
               do ib = 1, 2
                  ipart = -11
                  if (ia.eq.2) ipart = 11
                  ibeam = -11
                  if (ib.eq.2) ibeam = 11
                  k = k + 1
                  val(k) = eepdf_tilde(x(i), q2(i), n, ipart, ibeam)
               enddo
            enddo
         enddo
         write(*,'(18(1x,1p,e24.16))') x(i), q2(i), (val(k), k=1,16)
      enddo
      end
