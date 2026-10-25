      subroutine impose_frame_rest(ids, npart, p)
c**************************************************************************
c     Impose the defining property of a single-leg me_frame: if the frame
c     is built from exactly one selected leg, that leg is analytically at
c     rest after the boost, so force its three-momentum to exactly zero.
c
c     Why this is not cosmetic. boostx only reaches rest up to the rounding
c     of the boost factor, leaving a residual of a few 1d-14 whose
c     direction is pure noise. vxxxxx (aloha_functions.f) branches on
c     pp.eq.rZero and, for a massive vector at exactly zero three-momentum,
c     takes the frame z axis as quantisation axis -- the longitudinal
c     polarisation vector (0,0,0,1). Off that branch it builds the
c     polarisation vectors from the momentum direction, i.e. from the
c     noise, giving an O(1) wrong polarisation state on precisely the
c     events that fail to round to zero. Measured at 1.8% (5 sigma) on the
c     shipped LO polarised cross-section when this was first found.
c
c     Only nsel==1 needs it: with two or more selected legs it is their
c     *sum* that is at rest, no individual leg sits on the branch point,
c     and the residual of the sum never reaches HELAS.
c
c     The energy is deliberately left untouched. Zeroing the three-momentum
c     shifts the invariant mass by O(1d-28) relative, and HELAS takes the
c     mass as a separate argument anyway.
c
c     This lives in Template/Common/Source because it had to be got right
c     in four places -- Template/LO/SubProcesses/genps.f,
c     Template/NLO/SubProcesses/boost_to_frame.f and both copies in
c     MadSpin/src/driver.f -- and was for a long time got right in only
c     one. Keep it here rather than inlining it again.
c
c     input:  ids(npart)   0/1 mask of the legs the frame is built from
c             npart        number of legs in ids and p
c     in/out: p(0:3,npart) momenta, already boosted into the frame
c**************************************************************************
      implicit none
      integer npart
      integer ids(npart)
      double precision p(0:3,npart)
      integer i, nsel, isel

      nsel = 0
      isel = 0
      do i = 1, npart
         if (ids(i).eq.1) then
            nsel = nsel + 1
            isel = i
         endif
      enddo

      if (nsel.eq.1) then
         p(1,isel) = 0d0
         p(2,isel) = 0d0
         p(3,isel) = 0d0
      endif

      return
      end
