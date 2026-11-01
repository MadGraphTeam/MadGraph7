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
c
c     The frame kinematics proper (get_frame_mask_born/real,
c     get_me_frame_boost, boost_to_me_frame) are in me_frame_boost.f.
c
c     Each wrapper calls density_frame_rephase (density_weights.f) with the
c     momenta the matrix element gets: with density_nrk, the projector of
c     the density-matrix weights is turned into that of the {n,r,k} frame
c     of these momenta (a no-op otherwise, and without --density).
c**************************************************************************


      subroutine snap_mother_at_rest(ids, npart, imother, pm)
c**************************************************************************
c     When me_frame selects exactly one leg and that leg is the FKS mother,
c     the mother is analytically at rest in this frame. boostx only reaches
c     that to a few 1d-14, and everything downstream tests for rest with an
c     exact .eq.0d0 -- azifact_from_kperp divides by |pmother| and guards it
c     with rn.eq.0d0. Left as noise the guard never fires and the azimuth is
c     reconstructed from a direction that is pure rounding error; imposed,
c     the caller gets that routine's clear 'mother at rest' error instead of
c     a silently wrong phase. boost_to_me_frame imposes the same property on
c     the full momentum set, and for the same reason.
c**************************************************************************
      implicit none
      integer npart, imother
      integer ids(npart)
      double precision pm(0:3)
      integer i, nsel, isel

      nsel=0
      isel=0
      do i=1,npart
         if (ids(i).eq.1) then
            nsel=nsel+1
            isel=i
         endif
      enddo
      if (nsel.eq.1 .and. isel.eq.imother) then
         pm(1)=0d0
         pm(2)=0d0
         pm(3)=0d0
      endif

      return
      end


      subroutine azifact_me_frame(p_born_in, imother, p_i, p_j,
     &                            y_ij_fks, azifact, boosted)
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
      double precision p_i(0:3), p_j(0:3), y_ij_fks
      double complex azifact
      logical boosted
      double precision pboost(0:3), pm(0:3), kp(0:3)
      double precision pib(0:3), pjb(0:3)
      double precision cphi_m, sphi_m, vtiny
      parameter (vtiny=1d-8)
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
      call snap_mother_at_rest(ids, nexternal-1, imother, pm)

      if (1d0-y_ij_fks.lt.vtiny) then
c        Collinear: <ij>/[ij] is 0/0, so rebuild -exp(2 i psi) from the
c        stored emission direction. Same threshold as the legacy branch
c        in sborncol_fsr/isr, so boosted and unboosted switch together.
         call boostx(xij_kperp, pboost, kp)
         call azifact_from_kperp(pm, kp, azifact)
      else
c        Away from the collinear point the spinors are exact, while the
c        kperp reconstruction carries an O((kt/E)/betaT) error -- 136%
c        at betaT=1e-5 when this was measured for azifact_mc_frame,
c        which is why that routine branches the same way. sreal enters
c        here whenever 1-y_ij_fks < tiny (1e-6 in production), not only
c        at y=1, so this branch is reached with genuinely off-collinear
c        kinematics. Convert <ij>/[ij] to the -exp(2 i psi) this routine
c        contracts to return, using <ij>/[ij] = -exp(2 i psi) e^(2 i phi_m).
         call boostx(p_i, pboost, pib)
         call boostx(p_j, pboost, pjb)
         call azifact_from_spinors(pib, pjb, azifact)
         call getaziangles(pm, cphi_m, sphi_m)
         azifact=azifact*dcmplx(cphi_m,-sphi_m)**2
      endif
      boosted=.true.

      return
      end


      subroutine me_frame_born_boost(p_born_in, pboost, boosted)
c**************************************************************************
c     The boost sborn_frame() would apply to these Born momenta, exposed so
c     the MC counterterms can put their azimuthal ingredients in the same
c     frame. boosted=.false. means no frame was requested.
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      double precision p_born_in(0:3,nexternal-1), pboost(0:3)
      logical boosted
      integer ids(nexternal-1)
      logical trivial

      call get_frame_mask_born(ids)
      call get_me_frame_boost(p_born_in, nexternal-1, ids, pboost,
     &                        trivial)
      boosted=.not.trivial

      return
      end


      subroutine azifact_from_spinors(pi, pj, azifact)
c**************************************************************************
c     <ij>/[ij] from two massless momenta -- the expression the legacy
c     branches build inline. Those keep their own copies on purpose:
c     rewriting them would break the byte-identity of the no-frame path.
c**************************************************************************
      USE ALOHA_OBJECT
      implicit none
      double precision pi(0:3), pj(0:3)
      double complex azifact
      double precision zero
      parameter (zero=0d0)
      TYPE(ALOHA) W1,W2,W3,W4
      double complex Wij_angle,Wij_recta
      integer i

      CALL IXXXXX(pi ,ZERO ,+1,+1,1,W1)
      CALL OXXXXX(pj ,ZERO ,-1,+1,1,W2)
      CALL IXXXXX(pi ,ZERO ,-1,+1,1,W3)
      CALL OXXXXX(pj ,ZERO ,+1,+1,1,W4)
      Wij_angle=(0d0,0d0)
      Wij_recta=(0d0,0d0)
      do i=1,4
         Wij_angle = Wij_angle + W1%W(i)*W2%W(i)
         Wij_recta = Wij_recta + W3%W(i)*W4%W(i)
      enddo
      azifact=Wij_angle/Wij_recta

      return
      end


      subroutine azifact_mc_frame(p_born_in, imother, p_i, p_j,
     &                            y_ij_fks, pboost, azifact,
     &                            cphi_mother, sphi_mother)
c**************************************************************************
c     Azimuthal ingredients of the MC counterterms, in the me_frame, scaled
c     so that the caller's existing multiply comes out right:
c         ISR   -(cphi_mother + i sphi_mother)**2 * conjg(azifact)
c         FSR   -(cphi_mother - i sphi_mother)**2 *       azifact
c     both = +-exp(-+2 i psi), psi the emission azimuth about the mother.
c     The boost (pboost, from me_frame_born_boost) is the Born's own, since
c     the phase multiplies borntilde. azifact is <ij>/[ij] only in the
c     generic branch; the collinear one pre-cancels the caller's mother
c     factor instead, so do not assume it if you change the caller.
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      double precision p_born_in(0:3,nexternal-1)
      integer imother
      double precision p_i(0:3), p_j(0:3), y_ij_fks, pboost(0:3)
      double complex azifact
      double precision cphi_mother, sphi_mother
      double precision pm(0:3), kp(0:3), pib(0:3), pjb(0:3)
      integer ids(nexternal-1)
      double precision xij_kperp(0:3)
      common/cxij_kperp/xij_kperp

      call boostx(p_born_in(0,imother), pboost, pm)
      call get_frame_mask_born(ids)
      call snap_mother_at_rest(ids, nexternal-1, imother, pm)
      call getaziangles(pm, cphi_mother, sphi_mother)

      if (1d0-y_ij_fks.le.0d0) then
c        Exactly collinear: <ij>/[ij] is 0/0 in every frame, so rebuild it
c        from the stored emission direction, using
c            lim_{kt->0} <ij>/[ij] = -exp(2 i psi) * exp(2 i phi_m),
c        the identity that makes this branch agree with the generic one;
c        azifact_from_kperp returns only the first factor.
c
c        Taken at kt exactly zero and nowhere else, on purpose: the spinors
c        are exact at any kt>0, while this reconstruction carries an
c        O((kt/E)/betaT) error, betaT the transverse velocity of the boost.
         call boostx(xij_kperp, pboost, kp)
         call azifact_from_kperp(pm, kp, azifact)
         azifact=azifact*dcmplx(cphi_mother,sphi_mother)**2
      else
         call boostx(p_i, pboost, pib)
         call boostx(p_j, pboost, pjb)
         call azifact_from_spinors(pib, pjb, azifact)
      endif

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
      call density_frame_rephase(p_f, nexternal-1)
      call sborn(p_f, ans_summed)

      return
      end


      subroutine sborn_onehel_frame(p_in, hel1, chosen_hel, ans)
c**************************************************************************
c     One-helicity Born in the frame selected by me_frame.
c
c     Needed wherever a caller already goes through sborn_frame: SBORN
c     caches its amplitudes against (E, p_z) and SBORN_ONEHEL shares that
c     cache, so passing boosted momenta to one and unboosted to the other
c     silently returns amplitudes from the wrong frame -- see the note on
c     sborn_frame. check_sudakov does exactly that mix.
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      double precision p_in(0:3,nexternal-1)
      integer hel1, chosen_hel
      double precision ans
      double precision p_f(0:3,nexternal-1)
      integer ids(nexternal-1)

      call get_frame_mask_born(ids)
      call boost_to_me_frame(p_in, nexternal-1, ids, p_f)
      call density_frame_rephase(p_f, nexternal-1)
      call sborn_onehel(p_f, hel1, chosen_hel, ans)

      return
      end


      subroutine binothlha_frame(p_in, born_wgt, virt_wgt)
c**************************************************************************
c     Virtual in the me_frame rest frame.
c
c     BinothLHA hands its argument straight to sloopmatrix_thres, and takes
c     born_wgt from the caller rather than recomputing it, so boosting the
c     momenta here is enough to put the virtual in the same frame as the
c     Born that bornsoftvirtual already boosted.
c
c     Two MadLoop settings interact with an imposed frame, both via the
c     HELAS branch at exactly zero momentum that a one-leg me_frame sits on
c     (see boost_to_me_frame):
c
c       - NRotations_DP / NRotations_QP re-evaluate the loop at a rotated
c         phase-space point and expect |M|^2 back unchanged. That holds for
c         fixed helicities in general, but NOT for a particle at rest,
c         whose polarisation axis is the frame z axis rather than its own
c         momentum, so a rotation of the momenta does not carry the axis
c         with it. Both default to 0, so the rotation test is off unless a
c         user turns it on; if they do, a polarised run will see spurious
c         instability. Keep them at 0.
c       - ImprovePSPoint (default 2) shifts momenta to restore exact
c         onshellness, which can move the deliberately-zeroed leg off zero.
c         The shift is along z, so the momentum direction stays on the axis
c         and eps_L comes back along +-z, differing at most by a sign that
c         cancels in |M|^2. Checked to be harmless by check_poles rather
c         than assumed.
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      double precision p_in(0:3,nexternal-1)
      double precision born_wgt, virt_wgt
      double precision p_f(0:3,nexternal-1)
      integer ids(nexternal-1)

      call get_frame_mask_born(ids)
      call boost_to_me_frame(p_in, nexternal-1, ids, p_f)
      call density_frame_rephase(p_f, nexternal-1)
      call BinothLHA(p_f, born_wgt, virt_wgt)

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
      call density_frame_rephase(p_f, nexternal-1)
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
      call density_frame_rephase(p_f, nexternal-1)
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
      call density_frame_rephase(p_f, nexternal)
      call smatrix_real(p_f, wgt)

      return
      end
