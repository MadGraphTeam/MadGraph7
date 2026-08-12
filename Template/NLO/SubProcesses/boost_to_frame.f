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


      subroutine azifact_me_frame(p_born_in, imother, azifact, boosted)
c**************************************************************************
c     Collinear azimuthal factor -exp(2 i psi), evaluated in the me_frame.
c
c     The frame is rebuilt from the very momenta the matching Born call is
c     given, so the spin-correlated Born and the phase multiplying it are
c     guaranteed to live in the same frame. That is the whole point: the
c     Born is boosted inside sborn_frame, and if the phase were left in the
c     partonic c.m. the Q term would be wrong.
c
c     The mother and the stored emission direction are boosted together.
c     xij_kperp is orthogonal to the mother (k.p = 0 in the frame it was
c     built in), and orthogonality is preserved by the boost, so projecting
c     the boosted vector onto the helicity basis of the boosted mother
c     recovers the azimuth correctly.
c
c     boosted=.false. means no frame was requested (or the selection is one
c     of the partonic-c.m. spellings). azifact is then left untouched and
c     the caller must keep its legacy value, so that runs without a frame
c     stay bit-identical.
c
c     input:  p_born_in(0:3,nexternal-1)  Born momenta of this configuration
c             imother                     position of the mother in them
c     output: azifact, boosted
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      double precision p_born_in(0:3,nexternal-1)
      integer imother
      double complex azifact
      logical boosted
      double precision pboost(0:3), pm(0:3), kp(0:3)
      integer ids(nexternal-1)
      logical trivial
      double precision xij_kperp(0:3)
      common/cxij_kperp/xij_kperp

      call get_frame_mask_born(ids)
      call get_me_frame_boost(p_born_in, nexternal-1, ids, pboost,
     &                        trivial)
      if (trivial) then
         boosted=.false.
         return
      endif

      call boostx(p_born_in(0,imother), pboost, pm)
      call boostx(xij_kperp, pboost, kp)
      call azifact_from_kperp(pm, kp, azifact)
      boosted=.true.

      return
      end


      subroutine azifact_from_kperp(pmother, kperp, azifact)
c**************************************************************************
c     Rebuild the collinear limit of <ij>/[ij] from the emission direction.
c
c     The counterterms need -exp(2 i psi), with psi the azimuth of the
c     emission about the mother, measured in the HELAS transverse basis of
c     the mother direction. The generator stores that limit as xij_aor, but
c     xij_aor is a 0/0 limit of <ij>/[ij]: a boost maps exactly-parallel
c     null vectors onto exactly-parallel null vectors, so the degeneracy
c     survives in every frame and xij_aor can never be recomputed after a
c     boost. The azimuthal information has to be carried covariantly
c     instead, which is what xij_kperp is for.
c
c     Basis, for a mother direction n = (sin(th)cos(ph), sin(th)sin(ph),
c     cos(th)):
c         e1 = ( cos(th)cos(ph), cos(th)sin(ph), -sin(th) )
c         e2 = (        -sin(ph),      cos(ph),        0  )
c     which is the standard helicity basis; for n = +z it is (x,y) and for
c     n = -z it is (-x,y).
c
c     This reproduces both incoming legs with no idir bookkeeping:
c       j_fks=1, n=+z : psi = phi_i        -> -exp( 2 i phi_i)
c       j_fks=2, n=-z : psi = pi - phi_i   -> -exp(-2 i phi_i)
c     the doubled angle turning the basis flip into a harmless 2 pi. The
c     R_y(pi) rotation the old ISR code applied for j_fks=2 is therefore
c     not needed here -- and it must not be reinstated, since after a boost
c     it no longer maps the mother onto +z.
c
c     input:  pmother(0:3)  mother 4-momentum (only its direction is used)
c             kperp(0:3)    emission direction transverse to the mother
c     output: azifact       -exp(2 i psi)
c**************************************************************************
      implicit none
      double precision pmother(0:3), kperp(0:3)
      double complex azifact
      double precision n(3), e1(3), e2(3)
      double precision rn, cth, sth, cph, sph, a, b, norm2
      double complex ximag
      parameter (ximag=(0d0,1d0))
      integer i

      rn=sqrt(pmother(1)**2+pmother(2)**2+pmother(3)**2)
      if (rn.eq.0d0) then
         write (*,*) 'ERROR in azifact_from_kperp: mother at rest'
         stop 1
      endif
      do i=1,3
         n(i)=pmother(i)/rn
      enddo

      cth=n(3)
      sth=sqrt(max(1d0-cth**2,0d0))
      if (sth.ne.0d0) then
         cph=n(1)/sth
         sph=n(2)/sth
      else
c        mother on the beam axis: its azimuth is undefined, take phi=0.
c        Same convention as getaziangles().
         cph=1d0
         sph=0d0
      endif

      e1(1)= cth*cph
      e1(2)= cth*sph
      e1(3)=-sth
      e2(1)=-sph
      e2(2)= cph
      e2(3)= 0d0

      a=0d0
      b=0d0
      do i=1,3
         a=a+kperp(i)*e1(i)
         b=b+kperp(i)*e2(i)
      enddo

c     -exp(2 i psi) with psi=atan2(b,a), written without trigonometry so
c     that only the component of kperp transverse to the mother enters.
      norm2=a**2+b**2
      if (norm2.eq.0d0) then
         write (*,*) 'ERROR in azifact_from_kperp: the emission'
         write (*,*) 'direction is parallel to the mother, so its'
         write (*,*) 'azimuth is undefined.'
         stop 1
      endif
      azifact=-(dcmplx(a,b)**2)/norm2

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


      subroutine sborn_sf_frame(p_in, m, n, wgt)
c**************************************************************************
c     Colour-linked Born in the me_frame rest frame.
c
c     SBORN_SF reads the amplitudes SBORN left in the shared cache, so it
c     must be handed exactly the momenta the matching sborn_frame call was
c     handed, or it silently returns amplitudes from the other frame.
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      double precision p_in(0:3,nexternal-1)
      integer m, n
      double precision wgt
      double precision p_f(0:3,nexternal-1)
      integer ids(nexternal-1)

      call get_frame_mask_born(ids)
      call boost_to_me_frame(p_in, nexternal-1, ids, p_f)
      call sborn_sf(p_f, m, n, wgt)

      return
      end


      subroutine extra_cnt_frame(p_in, icnt, cnts)
c**************************************************************************
c     Extra g/a -> q qbar counterterm Born in the me_frame rest frame.
c     Same cache as SBORN, so the same rule applies.
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      double precision p_in(0:3,nexternal-1)
      integer icnt
      double complex cnts(2,nsplitorders)
      double precision p_f(0:3,nexternal-1)
      integer ids(nexternal-1)

      call get_frame_mask_born(ids)
      call boost_to_me_frame(p_in, nexternal-1, ids, p_f)
      call extra_cnt(p_f, icnt, cnts)

      return
      end


      subroutine smatrix_real_frame(p_in, wgt)
c**************************************************************************
c     Real-emission matrix element in the me_frame rest frame.
c
c     nexternal legs, so the mask is the real one; it is derived from the
c     Born mask and i_fks of the current FKS configuration.
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      double precision p_in(0:3,nexternal)
      double precision wgt
      double precision p_f(0:3,nexternal)
      integer ids(nexternal)
      integer nFKSprocess
      common/c_nFKSprocess/nFKSprocess

      call get_frame_mask_real(nFKSprocess, ids)
      call boost_to_me_frame(p_in, nexternal, ids, p_f)
      call smatrix_real(p_f, wgt)

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
      integer nsel, nini, nfin
      double precision m2, pvec2
      double precision betamin
      parameter (betamin=1d-4)
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

c     At rest to well below the precision the matrix elements can use:
c     skip too, because below betamin the boost costs more accuracy than
c     it delivers.
c
c     What it delivers is O(beta): that is how much a boost of velocity
c     beta moves a polarised |M|^2. What it costs is a loss of precision
c     in the incoming legs. Before the boost they are exactly on the beam
c     axis, so p(0)+p(3) of the backward one is exactly zero and the HELAS
c     massless spinors -- built from sqrt(p(0)+p(3)) and pt/sqrt(p(0)+p(3))
c     -- take their exact on-axis branch. The boost tilts them by O(beta),
c     making p(0)+p(3) = E*beta^2/2, and that is computed as a difference
c     of two numbers of size E, so it carries an absolute error ulp(E) and
c     a relative error 2*eps/beta^2. Small beta is the dangerous end.
c
c     Measured on p p > z{0} z{0} [real=QCD] with me_frame=[3,4], whose
c     boost degenerates to the identity as xi->0. Down the soft scan
c     p(0)+p(3) of the boosted incoming leg runs
c         3.3d-9, 3.3d-11, 3.4d-13, 0, 0, 0
c     and the deviation of the soft ratio from 1 runs
c         9d-6,   6.5d-4,  2.1d-2,  2d-6, 4d-7, 2d-6,
c     i.e. it grows like 1/beta^2 exactly while p(0)+p(3) is a few ulp,
c     and goes clean again once it underflows to exactly zero.
c
c     Balancing cost against benefit, 30*2*eps/beta^2 = beta (the factor 30
c     is the measured prefactor above), gives beta* = 2.4d-5. betamin is set
c     one safety decade above that. It was checked that 1d-5 is not enough
c     -- the split-order soft test still fails 0.16 -- and that 1d-4 and
c     1d-3 both give 0.01.
c
c     This costs no physics: it fires only where the selected system is at
c     rest to one part in 10^4, which in a real integration never happens.
c     It is reached only in the artificial deep-soft/collinear scan of
c     test_soft_col_limits, and there only for a frame that degenerates to
c     the partonic c.m. in the limit being scanned.
      if (pvec2 .lt. (betamin*pboost(0))**2) then
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
      integer i, nsel, isel

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
c     the boost has to be *exactly* right, not right to rounding.
c
c     HELAS switches convention at exactly zero: vxxxxx builds the
c     polarisation vectors of a massive vector along the z axis when
c     pp.eq.0d0, and along the momentum direction otherwise. boostx
c     reaches p=0 only up to a relative rounding of order 1d-16 -- it
c     forms p(i)+q(i)*lf with lf=1 up to the rounding of (q(0)-m)+p(0),
c     so the residual is a few 1d-14 in absolute value and its direction
c     is pure noise. When that residual happens not to round to zero,
c     eps_L points along the noise instead of along z and |M|^2 changes
c     by orders of magnitude.
c
c     Whether it rounds to zero is decided independently for the real
c     and for the reduced Born, so the two are then evaluated with
c     different quantisation axes and the subtraction stops cancelling.
c     Measured on p p > z{0} z{0} [real=QCD] with me_frame=[3]: the
c     soft/collinear ratio is 1 to 1d-7 whenever the residual is exactly
c     zero and 2.2d-3 whenever it is 3d-14, with no intermediate values.
c
c     Imposing the defining property of the frame removes the choice.
c     Only nsel=1 needs it: with two or more selected legs it is their
c     *sum* that is at rest, no individual leg sits at the branch point,
c     and the residual of the sum never reaches HELAS.
      nsel=0
      isel=0
      do i=1,npart
         if (ids(i).eq.1) then
            nsel=nsel+1
            isel=i
         endif
      enddo
      if (nsel.eq.1) then
         p_out(1,isel)=0d0
         p_out(2,isel)=0d0
         p_out(3,isel)=0d0
      endif

      return
      end
