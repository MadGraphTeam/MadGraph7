c**************************************************************************
c     Kinematics of the rest frame in which the matrix elements are
c     evaluated (me_frame in the run_card, frame_id in run.inc): which legs
c     define it (get_frame_mask_born/real) and the boost itself
c     (get_me_frame_boost, boost_to_me_frame).
c
c     Kept apart from boost_to_frame.f, which wraps the Born, real, virtual
c     and counterterm calls, because these routines call no matrix element:
c     the standalone Born driver (check_sa_fks.f) links them on their own.
c**************************************************************************

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
c     The answer depends only on frame_id, fixed for the run from the
c     run_card, and on frame_map_born, a compile-time DATA table -- but
c     this is called from every _frame wrapper on every phase-space point.
c     Cache it, keyed on frame_id so the cache cannot go stale if that
c     ever becomes settable mid-run. No OpenMP in these sources; MadEvent
c     parallelises by forking, so each process gets its own copy.
      integer cached_ids(nexternal-1)
      integer cached_id
      save cached_ids, cached_id
      data cached_id /-1/

      if (frame_id.ne.cached_id) then
         do i=1,nexternal-1
            cached_ids(i)=0
            if (btest(frame_id, frame_map_born(i))) then
               cached_ids(i)=1
            endif
         enddo
         cached_id=frame_id
      endif

      do i=1,nexternal-1
         ids(i)=cached_ids(i)
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
c     Note on r = j_fks: positionally the Born leg is the *mother* ij of
c     i_fks and j_fks. That is still the right leg to inherit the mask from,
c     and NOT because the polarised legs are spectators -- they are not: a
c     polarised massive quark IS j_fks in some configurations, and the real
c     carries its polarisation (see fks_common.carry_polarization).
c     It is right because in that case the real's j_fks is the SAME particle
c     as the Born's ij, so "the leg the user selected" means the same thing on
c     both sides, and the two matrix elements are boosted into frames that
c     coincide in the singular limit.
c
c     That equality is what makes the whole polarised subtraction well
c     defined, and it fails in two places, both refused at generation time by
c     madgraph_interface.check_process_format. Do not read this routine as a
c     licence to relax either refusal:
c       - a massless coloured emitter in the final state: g -> q q~ has no
c         leg with the mother's identity to correspond to.
c       - a coloured particle in the INITIAL state, whatever its mass. The
c         initial-state splitting runs backwards, g -> q(-> Born) q~, so the
c         polarised Born leg is an INTERNAL line of the real: there is no
c         real-side external leg for this mask to select at all, and the
c         positional r <-> b table above has nothing to say about it.
c
c     output: ids(nexternal)
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      include 'fks_info.inc'
      include 'nFKSconfigs.inc'
      include 'run.inc'
      integer iFKS
      integer ids(nexternal)
      integer ids_born(nexternal-1)
      integer r, b
c     Constant for a given iFKS: the Born mask is run-constant and the
c     r <-> b correspondence is fixed by fks_i_D(iFKS), a DATA table. So
c     cache one mask per FKS configuration rather than rebuilding both
c     this and the Born mask on every real-emission evaluation. Keyed on
c     frame_id for the same reason as get_frame_mask_born.
      integer cached_real(nexternal, fks_configs)
      logical cached_ok(fks_configs)
      integer cached_id_r
      save cached_real, cached_ok, cached_id_r
      data cached_ok /fks_configs*.false./
      data cached_id_r /-1/

      if (frame_id.ne.cached_id_r) then
         do r=1,fks_configs
            cached_ok(r)=.false.
         enddo
         cached_id_r=frame_id
      endif
      if (iFKS.ge.1 .and. iFKS.le.fks_configs) then
         if (cached_ok(iFKS)) then
            do r=1,nexternal
               ids(r)=cached_real(r,iFKS)
            enddo
            return
         endif
      endif

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

      if (iFKS.ge.1 .and. iFKS.le.fks_configs) then
         do r=1,nexternal
            cached_real(r,iFKS)=ids(r)
         enddo
         cached_ok(iFKS)=.true.
      endif

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
c     trivial is returned .true. when the boost should be skipped rather
c     than applied. Applying an identity boost is not free: it is a
c     multiply-and-add through boostx that perturbs the momenta in the last
c     bits, which is enough to move an adaptive integration. The LO code
c     avoids exactly this, in two places -- the call site skips frame_id=6
c     (auto_dsig_v4.inc) and boost_to_frame() skips the all-final-state
c     selection (genps.f) -- and this routine folds both in.
c
c     Four cases:
c       - nothing selected;
c       - the selection is exactly the initial state;
c       - the selection is exactly the whole final state;
c       - the selected system already has exactly zero 3-momentum.
c
c     The middle two are both "the partonic c.m.". Note they are *not*
c     identity boosts here the way they are at LO: MadFKS works in a frame
c     boosted along z (the two initial momenta carry different energies),
c     so honouring them literally would apply a real longitudinal boost. It
c     would also not be infrared safe -- a frame defined from the initial
c     state jumps between the real emission and the reduced Born, because
c     their momentum fractions differ by a finite amount even in the
c     singular limit. Skipping is both the safe and the meaningful choice.
c
c     input:  p(0:3,npart)  momenta of this configuration
c             ids(npart)    0/1 mask, bit n of frame_id = leg n+1
c     output: pboost(0:3)   boost 4-vector
c             trivial       .true. if the boost may be skipped
c**************************************************************************
      implicit none
      integer npart
      double precision p(0:3,npart), pboost(0:3)
      integer ids(npart)
      logical trivial
      integer i, j
      integer nsel, nini, nfin
      double precision m2, pvec2
      double precision m2i, betamin, mtiny
      parameter (betamin=1d-4, mtiny=1d-12)
      logical massive_ini
      include 'nexternal.inc'

      pboost(0:3)=0d0
      nsel=0
      nini=0
      nfin=0
      do i=1,npart
         if (ids(i).eq.1) then
            nsel=nsel+1
            if (i.le.nincoming) then
               nini=nini+1
            else
               nfin=nfin+1
            endif
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

c     Exactly the initial state, or exactly the whole final state: both name
c     the partonic c.m.. Skip rather than boost -- see the header.
c
c     Note the second of these is multiplicity-dependent and therefore does
c     not mean the same thing for the Born and for the real: me_frame=[3,4]
c     on p p > z z is the whole Born final state, but in the real the same
c     two legs are only two of three. It was checked that this asymmetry is
c     *not* what makes that configuration fail -- removing the skip leaves
c     the failure fractions bit-identical, because p_born is handed to us in
c     the Born partonic c.m. and so the zero-3-momentum test below catches
c     the same case anyway. The skip is kept for parity with LO.
      if (nfin.eq.0 .and. nini.eq.nincoming) then
         trivial=.true.
         return
      endif
      if (nini.eq.0 .and. nfin.eq.npart-nincoming) then
         trivial=.true.
         return
      endif

      pvec2=pboost(1)**2+pboost(2)**2+pboost(3)**2

c     Already at rest: skip, so the default costs nothing and changes nothing.
      if (pvec2.eq.0d0) then
         trivial=.true.
         return
      endif

c     A near-identity boost tilts the on-axis incoming partons just
c     enough that p(0)+p(3) of the backward one becomes E*beta**2/2,
c     computed as a difference of two numbers of size E: relative error
c     2*eps/beta**2, worst at small beta. A betamin=1d-4 skip used to
c     hide that (blocker B9), and was removed because PR #111 takes the
c     component from p+ p- = pT**2 instead, leaving no cancellation.
c
c     That holds for a MASSLESS incoming parton. PR #111 rewrote only
c     the fmass.eq.rZero branch of ixxxxx/oxxxxx; the massive branch
c     still forms pp3 = max(pp+p(3),rZero) and feeds it to
c     sqrt(pp3/(2*pp)), which is the same cancelling difference. So the
c     skip is kept exactly where it is still needed -- a massive
c     initial-state parton, e.g. b in a 4-flavour scheme -- and nowhere
c     else. The deeper fix is to give the massive branch the same
c     treatment as the massless one, across the ALOHA templates, the
c     legacy HELAS sources and the C++ backends; until that lands this
c     guard is what stops a massive beam leg losing its precision here.
      massive_ini=.false.
      do i=1,min(nincoming,npart)
         m2i=p(0,i)**2-p(1,i)**2-p(2,i)**2-p(3,i)**2
         if (m2i.gt.mtiny*p(0,i)**2) massive_ini=.true.
      enddo
      if (massive_ini .and. pvec2.lt.(betamin*pboost(0))**2) then
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

c     A selection made of a single leg puts that leg at rest, and there
c     the boost has to be *exactly* right, not right to rounding --
c     see Template/Common/Source/impose_frame_rest.f for why HELAS
c     cares.
c
c     What is specific to NLO is that whether the residual rounds to
c     zero is decided independently for the real and for the reduced
c     Born, so the two are then evaluated with different quantisation
c     axes and the subtraction stops cancelling. Measured on
c     p p > z{0} z{0} [real=QCD] with me_frame=[3]: the soft/collinear
c     ratio is 1 to 1d-7 whenever the residual is exactly zero and
c     2.2d-3 whenever it is 3d-14, with no intermediate values.
      call impose_frame_rest(ids, npart, p_out)

      return
      end
