! Copyright (C) 2026 CERN.
! Licensed under the GNU Lesser General Public License (version 3 or later).
! Created by A. Valassi (Aug 2026) for project MadVfold (arXiv:2609.23500).

! Road B prototype (eebb_vfalt) step 2: fixed-slot RNG for MINT0/MINT1.
! MINT2 is not touched yet, its call sites still use plain ran2().
! Design: every draw site below gets a fixed, named slot: this is drawn
! every time its enclosing scope (event or fold) is entered, regardless of
! whether the branch containing the original ran2() call is actually taken.
! This is what makes the draws' identity independent of which branches
! fire (an untaken branch just burns its slot instead of desyncing anything
! downstream), unlike plain sequential ran2() consumption.
!
! Mode behavior:
!   F0: vfold_newran2() delegates straight to ran2(): no slot involved, the
!       call site behaves exactly as it did before this module existed.
!   F, C: vfold_newran2() returns the pre-drawn value from the fixed slot
!       array for that site; the slot arrays are filled once per event
!       (vfold_fill_event_slots) and once per fold (vfold_fill_fold_slots),
!       called from mint_module.f90 (alongside get_random_x and
!       get_random_x_next_fold).
!
! NOT YET CONVERTED:
!   - fks_singular.f:1386 rndec(i) loop -- plotting-only, "will not change
!     between events", already behaves like a draw-once-and-cache pattern
!   - fks_singular.f pick_unweight_contr (3776, 3850) -- MINT2-only
!     (called from driver_mintMC.f:338, inside the imode.eq.2 block)
!   - driver_mintMC.f:322,1282, and mint_module.f90's get_ran and
!     get_weighted_cell -- MINT2-only (in gen(), not mint())
!     (check_upper_bound was converted in step 8q)
!   - setcuts.f:667 -- called once at program startup, not per-event
!     Step 14a: this was WRONG (sChan_order redraws at each new_point):
!     converted to ISLOT_EV_SCHAN_ORDER, see setcuts.f
!   - add_write_info.f fill_icolor_H (two g>gg colour-flow draws) -- not
!     in this list originally: also called from montecarlocounter.f
!     complete_xmcsubt (MC@NLO-Delta only) inside the walk.
!     Step 14b: converted there to ISLOT_FD_ICOLOR_H (one of the spare
!     fold slots), see vfold_newran2_icolor_h
!   - montecarlocounter.f assign_emsca_array's two draws are inside a loop
!     over (i,j) -- a single slot is only correct for the first iteration
!     (dead code if Delta is disabled: revisit before enabling Delta)
!     Road B step 11m: converted, see the Delta dipole slots below

module vfold_rng_slots
  use vfold_mode
  implicit none

  ! --- Event-level slots: drawn once per event (kpoint), reused across
  ! all folds of that event, exactly like get_random_x's own ndim draws. ---
  integer, parameter, public :: ISLOT_EV_MC_INTEGER       = 1 ! MC_integer.f:123
  integer, parameter, public :: ISLOT_EV_CHANNEL          = 2 ! mint_module.f90 get_channel
  integer, parameter, public :: ISLOT_EV_VIRTUAL_FRACTION = 3 ! fks_singular.f:7057 (gated to fold=0)
  integer, parameter, public :: ISLOT_EV_SHSC_FOLD_PICK   = 4 ! fks_singular.f:3432 (v1) and 3538 (v2) update_shower_scale_Sevents_v1/v2
  integer, parameter, public :: ISLOT_EV_SHSC_FKS_PICK_A  = 5 ! fks_singular.f:3561 (v2)
  integer, parameter, public :: ISLOT_EV_SHSC_FKS_PICK_B1 = 6 ! fks_singular.f:3580 (v2, mutually exclusive w/ A)
  integer, parameter, public :: ISLOT_EV_SHSC_FKS_PICK_B2 = 7 ! unused (spare)
  ! Step 8q: MINT2's Von Neumann accept/reject draw, as in step 9d (drawn with
  ! the candidate's other event slots, not as a bare ran3 after the batch).
  integer, parameter, public :: ISLOT_EV_UPPER_BOUND      = 8 ! mint_module.f90 check_upper_bound
  ! Step 14a: the random order of the s-channel invariants (setcuts.f
  ! sChan_order, once per kpoint: all its choices derive from this number).
  integer, parameter, public :: ISLOT_EV_SCHAN_ORDER      = 9 ! setcuts.f sChan_order
  integer, parameter, public :: NSLOT_EVENT = 9

  ! --- Fold-level slots: drawn once per fold. ---
  integer, parameter, public :: ISLOT_FD_GENMOM_COINFLIP  = 1 ! genps_fks.f:522
  integer, parameter, public :: ISLOT_FD_JPARTNER_1       = 2 ! montecarlocounter.f:2633
  integer, parameter, public :: ISLOT_FD_JPARTNER_2       = 3 ! montecarlocounter.f:2653
  integer, parameter, public :: ISLOT_FD_JPARTNER_3       = 4 ! montecarlocounter.f:2663
  integer, parameter, public :: ISLOT_FD_JPARTNER_4       = 5 ! montecarlocounter.f:2692
  integer, parameter, public :: ISLOT_FD_JPARTNER_LHE     = 6 ! montecarlocounter.f:2724 (ipartner_lhe fallback)
  integer, parameter, public :: ISLOT_FD_ASSIGN_EMSCA     = 7 ! montecarlocounter.f:4558
  integer, parameter, public :: ISLOT_FD_EMSCA_ARRAY_1    = 8 ! montecarlocounter.f:4651
  integer, parameter, public :: ISLOT_FD_EMSCA_ARRAY_2    = 9 ! montecarlocounter.f:4655
  integer, parameter, public :: ISLOT_FD_VIRT_HELICITY_PICK = 10 ! born_hel.f PICKHELICITYMC (called from BinothLHA when mc_hel=1) -- found 2026-09-02, was missing from the original audit entirely (born_hel.f wasn't in the file list checked)
  integer, parameter, public :: ISLOT_FD_VEGAS_X_BORN_1    = 11 ! driver_mintMC.f update_vegas_x, abrv='born' branch (i=nndim-2) -- dormant for eebb_vfalt's current abrv='all' config, converted preemptively 2026-09-02 per exhaustive ran2() audit
  integer, parameter, public :: ISLOT_FD_VEGAS_X_BORN_2    = 12 ! driver_mintMC.f update_vegas_x, abrv='born' branch (i=nndim-1)
  integer, parameter, public :: ISLOT_FD_VEGAS_X_BORN_3    = 13 ! driver_mintMC.f update_vegas_x, abrv='born' branch (i=nndim)
  ! Step 14b: fill_icolor_H when called from complete_xmcsubt (Delta), see
  ! vfold_newran2_icolor_h. This is the first of the three spare slots
  ! (14-16) that the original design kept "for anything missed in the first
  ! audit pass": they were already drawn for every fold, so using one does
  ! not change the random stream (nor any result of the other processes).
  ! As for all fold slots (e.g. ISLOT_FD_ASSIGN_EMSCA), all calls within one
  ! fold (repeated calls, all FKS configurations) read the same value. This
  ! is a choice of the vfold design, not stock behaviour: the stock code
  ! draws a fresh ran2() at every call. Each call of complete_xmcsubt uses
  ! at most one such draw (one g>gg branch) and each choice is still 50/50,
  ! so this should only correlate the choices within a fold, without bias
  ! (unlike the product of per-dipole Sudakov ratios, for which step 11m
  ! gave each Delta dipole its own slots). Not verified by a measurement.
  integer, parameter, public :: ISLOT_FD_ICOLOR_H         = 14 ! add_write_info.f fill_icolor_H (from complete_xmcsubt)
  integer, parameter, public :: NSLOT_FOLD = 16 ! spares for anything missed in the first audit pass (15-16 still unused)

  double precision, public, save :: event_slot(NSLOT_EVENT)
  double precision, public, save :: fold_slot(NSLOT_FOLD)

  ! --- Road B step 3: fold-history recording/replay ---
  ! A kpoint's folds are walked twice (gather momenta for batching, then
  ! finish weights once the batch ME is available) -- same two-walk
  ! structure the old two-pass architecture used (RANMAR save/restore),
  ! but slots make it much simpler: walk 2 just re-reads walk 1's frozen
  ! fold_slot values instead of replaying RANMAR. Event-level slots don't
  ! need this: they're already frozen for the whole kpoint by the time
  ! the fold loop starts, so walk 2 simply never re-fills them.
  ! Road B step 9l: 128 -> 4096. The capacity is per kpoint batch, so 128
  ! silently capped ENV_VFOLD_KPOINT_BATCH at 4 kpoints at folding 8,4,1, i.e.
  ! at most 128 events per bridge call, far too few to fill a GPU.
  ! Road B step 11g: 4096 -> 8192 (k256 at folding 8,4,1, for larger GPU batches).
  integer, parameter, public :: MAX_FOLDS_PROTO = 8192
  ! Road B step 9b+9c: one fold history per MINT2 candidate queue (one queue
  ! per vn), selected by vfold_fold_qidx (1 for MINT0/MINT1 and unbatched).
  integer, parameter, public :: VFOLD_NVN = 3 ! vn = 1 virt, 2 novi, 3 born
  double precision, public, save :: &
       fold_slot_history(NSLOT_FOLD, MAX_FOLDS_PROTO, VFOLD_NVN)
  integer, public, save :: vfold_fold_qidx = 1
  logical, public, save :: vfold_replay_mode = .false.
  integer, public, save :: vfold_fold_idx = 0
  ! Road B step 9b+9c: replay the frozen fold slots (vfold_fold_replay) without
  ! necessarily retrieving batched MEs (vfold_replay_mode). F1/F2 replay the
  ! slots of a pre-generated MINT2 candidate but evaluate its MEs directly.
  logical, public, save :: vfold_fold_replay = .false.
  ! Step 14b: set by complete_xmcsubt (montecarlocounter.f) around its call
  ! of fill_icolor_H (add_write_info.f), which is inside the walk; the other
  ! call of fill_icolor_H, when writing an accepted event, keeps ran2().
  logical, public, save :: vfold_icolor_h_in_walk = .false.

  ! --- Road B step 8q: independent RANMAR streams (as in step 9d) ---
  ! Stream 1 is the original RANMAR stream (MINT0/MINT1 and all per-event
  ! draws in driver_mintMC.f); stream 1+vn is used only by the MINT2
  ! candidates of that vn. In step 9d this makes events.lhe independent of
  ! the kpoint batch size; here it is meant to give 8q the same numbers as 9d.
  ! A stream is a saved copy of RANMAR's /raset1/ and /raset2/ commons.
  ! F0 never switches streams.
  integer, parameter, public :: VFOLD_NSTREAM = 1 + VFOLD_NVN
  double precision, save :: st_ranu(97,VFOLD_NSTREAM)
  double precision, save :: st_ranc(VFOLD_NSTREAM),st_rancd(VFOLD_NSTREAM),st_rancm(VFOLD_NSTREAM)
  integer, save :: st_iranmr(VFOLD_NSTREAM),st_jranmr(VFOLD_NSTREAM)
  logical, save :: st_valid(VFOLD_NSTREAM) = .false.
  integer, public, save :: vfold_cur_stream = 1

  ! --- Road B step 9: EVENT (kpoint) batching ---
  ! When several kpoints are gathered before any is replayed, the next
  ! kpoint's vfold_fill_event_slots overwrites event_slot, so each kpoint's copy
  ! is kept (as fold slots already are).
  integer, parameter, public :: MAX_KPOINTS_PROTO = 8192 ! Road B step 11g (was 4096)
  double precision, public, save :: &
       event_slot_history(NSLOT_EVENT, MAX_KPOINTS_PROTO, VFOLD_NVN)
  ! Set while gathering a batch of kpoints: the batch driver, not
  ! get_random_x's vfold_reset_fold_idx, owns vfold_fold_idx, so the kpoints use
  ! successive slices of fold_slot_history (each records its start offset).
  logical, public, save :: vfold_kp_hold_fold_idx = .false.

  ! --- Road B step 11m: MC@NLO-Delta dipole slots ---
  ! montecarlocounter.f assign_emsca_array draws two numbers for each (i,j)
  ! dipole (stock: two ran2() per dipole per call). One fold slot pair for all
  ! dipoles would make the Delta starting scales of all dipoles fully
  ! correlated, so each dipole gets its own pair of fold slots. The repeated
  ! calls within one fold read the same values (as for ISLOT_FD_ASSIGN_EMSCA).
  ! Allocated and filled only if mcatnlo_delta is set (vfold_delta_slots_init),
  ! so that the random numbers of non-Delta runs do not change.
  integer, public, save :: vfold_delta_ndip = 0 ! nexternal-1 (0: Delta slots off)
  integer, save :: vfold_delta_nslot = 0 ! 2*ndip*ndip
  double precision, allocatable, save :: fold_slot_delta(:)
  double precision, allocatable, save :: fold_slot_delta_history(:,:,:)

contains

  subroutine vfold_rng_save(is)
    implicit none
    integer, intent(in) :: is
    double precision :: ranu(97),ranc,rancd,rancm
    common / raset1 / ranu,ranc,rancd,rancm
    integer :: iranmr,jranmr
    common / raset2 / iranmr,jranmr
    st_ranu(1:97,is)=ranu(1:97)
    st_ranc(is)=ranc
    st_rancd(is)=rancd
    st_rancm(is)=rancm
    st_iranmr(is)=iranmr
    st_jranmr(is)=jranmr
  end subroutine vfold_rng_save

  subroutine vfold_rng_load(is)
    implicit none
    integer, intent(in) :: is
    double precision :: ranu(97),ranc,rancd,rancm
    common / raset1 / ranu,ranc,rancd,rancm
    integer :: iranmr,jranmr
    common / raset2 / iranmr,jranmr
    ranu(1:97)=st_ranu(1:97,is)
    ranc=st_ranc(is)
    rancd=st_rancd(is)
    rancm=st_rancm(is)
    iranmr=st_iranmr(is)
    jranmr=st_jranmr(is)
  end subroutine vfold_rng_load

  ! Seed a stream lazily on first use, from the live RANMAR state, without
  ! drawing from stream 1 (an eager ran2() seeding perturbed stream 1 and
  ! broke ntuple's one-time initialisation).
  subroutine vfold_rng_seed_stream(is)
    implicit none
    integer, intent(in) :: is
    integer :: ij,kl
    double precision :: ranu(97),ranc,rancd,rancm
    common / raset1 / ranu,ranc,rancd,rancm
    integer :: iranmr,jranmr
    common / raset2 / iranmr,jranmr
    call vfold_rng_save(1) ! keep the caller's (stream 1) state
    ij = mod(abs(int(ranu(1)*30011d0)) + is*7919, 31329)
    kl = mod(abs(int(ranu(2)*30011d0)) + is*4643, 30082)
    call rmarin(ij,kl) ! overwrites the common block
    call vfold_rng_save(is)
    call vfold_rng_load(1) ! put the caller's stream back
    st_valid(is) = .true.
  end subroutine vfold_rng_seed_stream

  subroutine vfold_rng_use_stream(is)
    implicit none
    integer, intent(in) :: is
    if (.not. vfold_use_rng_slots()) return ! F0: untouched baseline
    if (is.eq.vfold_cur_stream) return
    if (is.lt.1 .or. is.gt.VFOLD_NSTREAM) then
      write(6,*) 'ERROR vfold_rng_use_stream: bad stream ',is
      stop 1
    endif
    ! Never seed stream 1: it is the original RANMAR stream.
    if (is.gt.1 .and. .not.st_valid(is)) call vfold_rng_seed_stream(is)
    call vfold_rng_save(vfold_cur_stream)
    call vfold_rng_load(is)
    vfold_cur_stream = is
  end subroutine vfold_rng_use_stream


  subroutine vfold_reset_fold_idx
    implicit none
    if (vfold_kp_hold_fold_idx) return ! kpoint-batched gathering owns vfold_fold_idx
    vfold_fold_idx = 0
  end subroutine vfold_reset_fold_idx

  ! Seek vfold_fold_idx to the start of one kpoint's slice of fold_slot_history,
  ! for its replay walk (offset 0 is vfold_reset_fold_idx, so the unbatched
  ! path is unchanged).
  subroutine vfold_set_fold_idx(ifold_start)
    implicit none
    integer, intent(in) :: ifold_start
    vfold_fold_idx = ifold_start
  end subroutine vfold_set_fold_idx

  subroutine vfold_save_event_slots(ikp)
    implicit none
    integer, intent(in) :: ikp
    if (.not. vfold_use_rng_slots()) return
    if (ikp.lt.1 .or. ikp.gt.MAX_KPOINTS_PROTO) then
      write(6,*) 'ERROR vfold_save_event_slots: ikp out of range', ikp, &
                 ' MAX_KPOINTS_PROTO=', MAX_KPOINTS_PROTO
      stop 1
    endif
    event_slot_history(1:NSLOT_EVENT,ikp,vfold_fold_qidx) = event_slot(1:NSLOT_EVENT)
  end subroutine vfold_save_event_slots

  subroutine vfold_restore_event_slots(ikp)
    implicit none
    integer, intent(in) :: ikp
    if (.not. vfold_use_rng_slots()) return
    event_slot(1:NSLOT_EVENT) = event_slot_history(1:NSLOT_EVENT,ikp,vfold_fold_qidx)
  end subroutine vfold_restore_event_slots

  ! Called once per event (kpoint), right after/alongside get_random_x.
  ! No-op in F0 mode: F0 must never consume an extra RANMAR draw the
  ! original code didn't already expect.
  subroutine vfold_fill_event_slots
    implicit none
    double precision, external :: ran2
    integer :: i
    if (.not. vfold_use_rng_slots()) return
    do i=1,NSLOT_EVENT
      event_slot(i) = ran2()
    enddo
  end subroutine vfold_fill_event_slots

  ! Called once per fold, right after/alongside get_random_x_next_fold.
  ! Road B step 3: when vfold_fold_replay is set (walk 2 of a kpoint, after
  ! walk 1 already gathered momenta for batching), re-read walk 1's frozen
  ! values for this fold index instead of drawing fresh ones -- walk 2
  ! must reproduce walk 1's momenta exactly, or the batched ME (evaluated
  ! from walk 1's momenta) would not correspond to walk 2's kinematics.
  subroutine vfold_fill_fold_slots
    implicit none
    double precision, external :: ran2
    integer :: i
    if (.not. vfold_use_rng_slots()) return
    vfold_fold_idx = vfold_fold_idx + 1
    if (vfold_fold_idx.gt.MAX_FOLDS_PROTO) then
      write(6,*) 'ERROR vfold_fill_fold_slots: vfold_fold_idx exceeds ', &
                 'MAX_FOLDS_PROTO=', MAX_FOLDS_PROTO
      stop 1
    endif
    if (vfold_fold_replay) then
      fold_slot(1:NSLOT_FOLD) = fold_slot_history(1:NSLOT_FOLD,vfold_fold_idx,vfold_fold_qidx)
    else
      do i=1,NSLOT_FOLD
        fold_slot(i) = ran2()
      enddo
      fold_slot_history(1:NSLOT_FOLD,vfold_fold_idx,vfold_fold_qidx) = fold_slot(1:NSLOT_FOLD)
    endif
    ! Road B step 11m: Delta dipole slots (after the standard ones)
    if (vfold_delta_nslot.gt.0) then
      if (vfold_fold_replay) then
        fold_slot_delta(:) = fold_slot_delta_history(:,vfold_fold_idx,vfold_fold_qidx)
      else
        do i=1,vfold_delta_nslot
          fold_slot_delta(i) = ran2()
        enddo
        fold_slot_delta_history(:,vfold_fold_idx,vfold_fold_qidx) = fold_slot_delta(:)
      endif
    endif
  end subroutine vfold_fill_fold_slots

  ! Road B step 11m: enable the Delta dipole slots (call once, after setrun,
  ! only if mcatnlo_delta is set). No-op in F0 (plain ran2() as in stock).
  subroutine vfold_delta_slots_init(ndip)
    implicit none
    integer, intent(in) :: ndip
    if (.not. vfold_use_rng_slots()) return
    if (vfold_delta_nslot.gt.0) return
    vfold_delta_ndip = ndip
    vfold_delta_nslot = 2*ndip*ndip
    allocate(fold_slot_delta(vfold_delta_nslot))
    allocate(fold_slot_delta_history(vfold_delta_nslot, MAX_FOLDS_PROTO, VFOLD_NVN))
    fold_slot_delta = 0d0
    fold_slot_delta_history = 0d0
  end subroutine vfold_delta_slots_init

  ! Road B step 11m: drop-in replacement for the two bare ran2() calls of
  ! dipole (i,j) in assign_emsca_array (k=1,2).
  double precision function vfold_newran2_delta(k, i, j)
    implicit none
    integer, intent(in) :: k, i, j
    double precision, external :: ran2
    if (.not. vfold_use_rng_slots() .or. vfold_delta_nslot.eq.0) then
      vfold_newran2_delta = ran2() ! F0, or a program without the slots (e.g. test_MC)
    else
      if (k.lt.1 .or. k.gt.2 .or. &
          i.lt.1 .or. i.gt.vfold_delta_ndip .or. j.lt.1 .or. j.gt.vfold_delta_ndip) then
        write(6,*) 'ERROR vfold_newran2_delta: bad slot', k, i, j, vfold_delta_ndip
        stop 1
      endif
      vfold_newran2_delta = fold_slot_delta(k + 2*((i-1) + vfold_delta_ndip*(j-1)))
    endif
  end function vfold_newran2_delta

  ! Step 14b: drop-in replacement for the bare ran2() calls of fill_icolor_H
  ! (add_write_info.f): the fold slot ISLOT_FD_ICOLOR_H if called inside the
  ! walk (from complete_xmcsubt), plain ran2() otherwise (event writing, F0).
  double precision function vfold_newran2_icolor_h()
    implicit none
    double precision, external :: ran2
    if (vfold_icolor_h_in_walk) then
      vfold_newran2_icolor_h = vfold_newran2(ISLOT_FD_ICOLOR_H, .true.)
    else
      vfold_newran2_icolor_h = ran2()
    endif
  end function vfold_newran2_icolor_h

  ! Drop-in replacement for a bare ran2() call; is_fold selects which slot
  ! array islot indexes into (ignored in F0 mode).
  double precision function vfold_newran2(islot, is_fold)
    implicit none
    integer, intent(in) :: islot
    logical, intent(in) :: is_fold
    double precision, external :: ran2
    if (.not. vfold_use_rng_slots()) then
      vfold_newran2 = ran2()
    elseif (is_fold) then
      vfold_newran2 = fold_slot(islot)
    else
      vfold_newran2 = event_slot(islot)
    endif
  end function vfold_newran2


  ! Road B step 9l: the capacities as compiled into THIS object file, for
  ! vfold_check_protos (mint_module.f90), which detects objects compiled
  ! against different values (e.g. a stale object returned by f90cache,
  ! whose key does not include the contents of the .mod files used).
  integer function vfold_rng_slots_proto(i)
    implicit none
    integer, intent(in) :: i
    vfold_rng_slots_proto = MAX_FOLDS_PROTO
    if (i.eq.2) vfold_rng_slots_proto = MAX_KPOINTS_PROTO
  end function vfold_rng_slots_proto
end module vfold_rng_slots
