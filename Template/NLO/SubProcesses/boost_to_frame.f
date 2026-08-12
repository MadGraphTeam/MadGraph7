c**************************************************************************
c     Rest frame in which the matrix-elements are evaluated.
c
c     Polarisation of a massive particle is not boost invariant, so for a
c     polarised process the matrix-elements must be evaluated in a definite
c     frame, chosen by the user through me_frame in the run_card. This file
c     provides the kinematics; the selection of which legs define the frame
c     is resolved by the caller and passed in as the mask ids().
c
c     IMPORTANT (see docs/nlo_polarisation_boost_plan.md, D3): the frame must
c     be recomputed from the momenta of *each* kinematic configuration (Born,
c     real, and every counter-event separately). Computing it once from the
c     real and reusing it for the reduced Born breaks the cancellation of the
c     collinear pole.
c**************************************************************************

      subroutine mapid_frame(id, npart, ids)
c**************************************************************************
c     Uncompress frame_id into a 0/1 mask over npart legs.
c     Bit n of id is set for each leg n selected by me_frame. This is the
c     same encoding as mapid() in the LO cluster.f -- note it is 2**n, not
c     2**(n-1), whatever the stale comment above the LO boost_to_frame says.
c
c     input:  id      compressed frame id (run_card frame_id)
c             npart   number of legs of this configuration
c     output: ids     ids(i)=1 if leg i takes part in the frame definition
c**************************************************************************
      implicit none
      integer id, npart, ids(npart)
      integer i

      do i=1,npart
         ids(i)=0
         if (btest(id,i)) then
            ids(i)=1
         endif
      enddo

      return
      end


      subroutine sborn_frame(p_in, ans_summed)
c**************************************************************************
c     Evaluate the Born in the frame selected by me_frame.
c
c     The frame is rebuilt from the momenta actually passed in, so each
c     kinematic configuration (the Born of the n-body contribution, and in
c     later milestones each counter-event separately) gets its own boost.
c     Reusing one frame across configurations breaks the cancellation of the
c     collinear pole -- see docs/nlo_polarisation_boost_plan.md, D3.
c
c     The boost is applied to a local copy: /pborn/ is read by the event
c     record, the cuts and the scales, none of which want the ME frame.
c
c     Every caller of SBORN within one event must agree on whether it passes
c     boosted or unboosted momenta: SBORN caches its amplitudes against
c     (E, p_z) and that cache is shared with sborn_sf, born_hel and
c     extra_cnt. Mixing the two silently returns amplitudes from the wrong
c     frame.
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      double precision p_in(0:3,nexternal-1)
      double precision ans_summed
      double precision p_f(0:3,nexternal-1)
      integer ids(nexternal-1)

      call get_frame_mask_born(ids)
      call boost_to_me_frame(p_in, nexternal-1, ids, p_f)
      call sborn(p_f, ans_summed)

      return
      end


      subroutine get_frame_mask_born(ids)
c**************************************************************************
c     Mask over the nexternal-1 Born legs selected by me_frame.
c
c     me_frame is given in the numbering of the process as the user wrote
c     it, which the FKS sort does not preserve; frame_map_born (written at
c     generation time into frame_info.inc) maps Born position -> user number.
c
c     output: ids(nexternal-1)
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      include 'frame_info.inc'
      include 'run.inc'
      integer ids(nexternal-1)
      integer i

      do i=1,nexternal-1
         ids(i)=0
         if (btest(frame_id, frame_map_born(i))) then
            ids(i)=1
         endif
      enddo

      return
      end


      subroutine get_frame_mask_real(iFKS, ids)
c**************************************************************************
c     Mask over the nexternal legs of the real emission selected by
c     me_frame, for FKS configuration iFKS.
c
c     The FKS convention fixes the real -> underlying-Born correspondence in
c     terms of i_fks alone (see set_pdg in chooser_functions.f):
c         real r  ->  Born r      for r <  i_fks
c         real r  ->  Born r-1    for r >  i_fks
c         real i_fks has no Born counterpart (it is the extra parton)
c     so no separate table is needed for the reals.
c
c     Note on r = j_fks: positionally the Born leg is the *mother* of i_fks
c     and j_fks, not the same particle as the real leg j_fks. Selecting a
c     splitting parton in me_frame is meaningless for polarisation anyway --
c     the particles whose polarisation is being fixed are spectators, present
c     unchanged in both the real and the Born.
c
c     output: ids(nexternal)
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      include 'fks_info.inc'
      integer iFKS
      integer ids(nexternal)
      integer ids_born(nexternal-1)
      integer r, b

      call get_frame_mask_born(ids_born)

      do r=1,nexternal
         ids(r)=0
         if (r.eq.fks_i_D(iFKS)) then
c           the extra parton: no Born counterpart, never part of the frame
            cycle
         elseif (r.lt.fks_i_D(iFKS)) then
            b=r
         else
            b=r-1
         endif
         if (b.ge.1 .and. b.le.nexternal-1) then
            ids(r)=ids_born(b)
         endif
      enddo

      return
      end


      subroutine get_me_frame_boost(p, npart, ids, pboost, trivial)
c**************************************************************************
c     Build the boost 4-vector that takes the selected system to rest.
c
c     pboost is (E, -Pvec) of the sum of the selected momenta, i.e. exactly
c     the argument boostx() wants in order to express a momentum given in
c     the current frame in the rest frame of that system.
c
c     trivial is returned .true. when the boost is the identity by
c     construction, so that callers can skip it and stay bit-identical to a
c     run with no frame selection at all. Two cases:
c       - nothing selected;
c       - the selected system has exactly zero 3-momentum (the default
c         me_frame=[1,2] in the partonic c.m., where MadFKS already works).
c
c     input:  p(0:3,npart)  momenta of this configuration
c             ids(npart)    0/1 mask, see mapid_frame
c     output: pboost(0:3)   boost 4-vector
c             trivial       .true. if the boost may be skipped
c**************************************************************************
      implicit none
      integer npart
      double precision p(0:3,npart), pboost(0:3)
      integer ids(npart)
      logical trivial
      integer i, j
      integer nsel
      double precision m2, pvec2

      pboost(0:3)=0d0
      nsel=0
      do i=1,npart
         if (ids(i).eq.1) then
            nsel=nsel+1
            do j=0,3
               pboost(j)=pboost(j)+p(j,i)
            enddo
         endif
      enddo

c     Nothing selected: nothing to do. Do not treat this as an error, it is
c     how frame_id=0 (or a mask that misses this configuration) shows up.
      if (nsel.eq.0) then
         trivial=.true.
         return
      endif

      pvec2=pboost(1)**2+pboost(2)**2+pboost(3)**2

c     Already at rest: skip, so the default costs nothing and changes nothing.
      if (pvec2.eq.0d0) then
         trivial=.true.
         return
      endif

c     The selected system must be timelike for its rest frame to exist. A
c     lightlike or spacelike sum means the run_card selection is nonsense
c     (e.g. a single massless leg); the LO boost_to_frame has no such guard.
      m2=pboost(0)**2-pvec2
      if (m2.le.0d0) then
         write (*,*) 'ERROR in get_me_frame_boost: the system'
         write (*,*) 'selected by me_frame is not timelike, so its'
         write (*,*) 'rest frame does not exist. m^2 = ', m2
         write (*,*) 'Selected legs:', (ids(i),i=1,npart)
         stop 1
      endif

      do j=1,3
         pboost(j)=-pboost(j)
      enddo
      trivial=.false.

      return
      end


      subroutine boost_to_me_frame(p, npart, ids, p_out)
c**************************************************************************
c     Express all npart momenta in the rest frame of the selected system.
c     Pure boost, no rotation: for fixed external helicities |M|^2 is
c     rotation invariant (HELAS builds the polarisation vectors in each
c     particle's own momentum-direction basis, so a rotation only produces
c     per-particle phases which cancel in the modulus). Orientation does
c     matter for the collinear counterterm azimuthal phases, but that is
c     handled by the caller, not here.
c
c     p and p_out may be the same array.
c**************************************************************************
      implicit none
      integer npart
      double precision p(0:3,npart), p_out(0:3,npart)
      integer ids(npart)
      double precision pboost(0:3)
      logical trivial
      integer i

      call get_me_frame_boost(p, npart, ids, pboost, trivial)

      if (trivial) then
         do i=1,npart
            p_out(0:3,i)=p(0:3,i)
         enddo
         return
      endif

      do i=1,npart
         call boostx(p(0,i), pboost, p_out(0,i))
      enddo

      return
      end
