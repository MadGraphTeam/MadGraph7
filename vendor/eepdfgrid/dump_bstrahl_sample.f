c     Reference for madspace's BeamstrahlungSampler: eepdf_sample_bstrahl of
c     bstrahl.f, compiled with a set's bstrahl_pars.inc, on given
c     (x, Q2, component, beamid, u1, u2).
c
c     The C++ port reproduces this to 1e-16 for all three collider q-term
c     forms.  Kept because that comparison is the only thing that pins the
c     port down -- the unit tests check invariants, not the spectrum itself.
c
c     bstrahl.f must sit next to the set's include (gfortran looks there
c     first, where Source/PDF keeps the no-beamstrahlung default):
c     cp Template/Common/Source/PDF/bstrahl.f \
c        Template/Common/Source/PDF/lep_densities/<set>/bstrahl_pars.inc w/
c     gfortran -w -ffixed-line-length-132 -o dump_bs dump_bstrahl_sample.f \
c         w/bstrahl.f
c     ./dump_bs points.txt      # npt, then "x q2 component beamid u1 u2"

      program dump_bs
      implicit none
      double precision x, q2, u1, u2, xbs, xisr
      integer npt, i, n, beamid
      character*256 fname
      call getarg(1, fname)
      open(unit=11, file=fname, status='old')
      read(11,*) npt
      do i = 1, npt
         read(11,*) x, q2, n, beamid, u1, u2
         call eepdf_sample_bstrahl(x, q2, n, beamid, beamid, u1, u2,
     &                             xbs, xisr)
         write(*,'(2(1x,1p,e27.16e4))') xbs, xisr
      enddo
      end
