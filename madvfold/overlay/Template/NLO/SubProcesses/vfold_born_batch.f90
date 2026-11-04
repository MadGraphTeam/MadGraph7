! Copyright (C) 2026 CERN.
! Licensed under the GNU Lesser General Public License (version 3 or later).
! Created by A. Valassi (Aug 2026) for project MadVFold (arXiv:2609.23500).
!-------------------------------------------------------------------------
! Road B prototype (eebb_vfalt) step 3. Born batching.
! Batch/cache mechanism for deferring SBORN evaluation, now wired into
! compute_born (fks_singular.f) via a two-walk-per-kpoint structure in
! compute_integrand (mint_module.f90) -- see the comment block there.
!
! Design principle (re-derived from how the OLD two-pass architecture on
! eebb_vfold actually worked, not assumed): add_wgt depends on a lot of
! ambient state set up elsewhere in sigintF's per-fold flow (Bjorken x's,
! scales, boost, FKS config, H/S-event flag, the weight_lines module's own
! ordering-dependent icontr counter). None of that is cached here. Instead,
! walk 2 re-runs sigintF's entire per-fold flow a second time, exactly
! like the old collection/consumption passes did -- the only thing walk 2
! does NOT recompute is the SBORN call itself (and its two side-effect
! outputs, amp_split and wgt_ME_born), which it retrieves from this
! module's cache instead. Slots (vfold_rng_slots) replace RANMAR save/restore
! as the mechanism keeping walk 2 perfectly synchronized with walk 1.
!
! What IS cached, and why each piece is safe to cache (small, local,
! read only by compute_born's own iamp loop -- never touched by add_wgt
! or anything else):
!   p_born     : the momenta themselves, obviously needed for SBORN.
!   f_b, g     : read by compute_born's iamp loop (wgt1=amp_split(iamp)
!                *f_b/g**qcd_power) -- NOT touched by SBORN itself, so
!                these must be captured at stash time, before they can
!                change for a later fold.
!   amp_split, wgt_ME_born : SBORN's own two side-effect outputs (via
!                /TO_AMP_SPLIT/ and /C_WGT_ME_TREE/ respectively).
!-------------------------------------------------------------------------
! Road B step 5. Born batching and cudacpp offloading.
! Now vfold_evaluate_born_batch has two different implementations for modes
! F and C, selected by vfold_use_cudacpp() (vfold_mode.f90).
! Mode F keeps calling SBORN in a plain Fortran loop, as in step 3.
! Mode C instead calls the cudacpp Born bridge (vfold_cudacpp_dispatcher.cpp,
! dlopen-loading a library in lib/cudacpp/build.<backend>_d_inl0_hrd0)
! once per kpoint, evaluating all VECSIZE_BORN_PROTO slots in one vector
! call -- unused slots (born_batch_count+1..VECSIZE_BORN_PROTO) are padded
! by repeating the last real entry's momenta/g (matches eebb_vfold's own
! chunk-padding technique) and their results are simply never retrieved.
! Unlike SBORN, the bridge has no amp_split-by-coupling-order breakdown:
! it returns one combined ME per event, reconstructed into amp_split
! exactly like eebb_vfold's own compute_born bridge branch does
! (amp_split(1)=mes, everything else 0 -- correct because Born only ever
! populates the single QCD^0*QED^4 slot, AMP_SPLIT_SIZE_BORN=1). Unlike
! the reference, wgt_ME_born is also explicitly set here from the same
! value (the reference leaves it at whatever SBORN last set it to
! elsewhere, since the bridge branch never touches it) -- done here so
! that mode C stays bit-identical to mode F for every value this module
! hands back, not just the ones the iamp loop directly consumes.
!-------------------------------------------------------------------------
! Road B step 6b. Bug fix in Real batching and cudacpp offloading.
! Bug found 2026-09-02, debugging an eebbbb F-vs-C disagreement: mode F's
! Fortran loop below did not set g=g_batch(i) nor call update_as_param()
! before each SBORN call, unlike vfold_real_batch.f90's Fortran loop (which
! always did both, with the same rationale -- SBORN's HELAS calls read
! derived couplings in coupl.inc's /couplings/ common, not raw g
! directly). Without the refresh, SBORN silently used whatever ambient
! coupling state was last left behind by something else (e.g. Real's own
! gathering activity earlier in the same walk), not this batch's own g.
! This is invisible for a process whose Born has zero alpha_s dependence
! (e.g. eebb, BORN_ORDERS QCD=0) but a silent, systematic mis-evaluation
! for a process whose Born genuinely depends on alpha_s (e.g. eebbbb,
! via t-channel gluon exchange between its two b bbar pairs).
!-------------------------------------------------------------------------
! Road B step 7b: the cudacpp bridge is called in chunks of
! vfold_vecsize_used events (ENV_VFOLD_VECSIZE_USED, default 128), padding only
! the last chunk. Previously every call evaluated all 128 slots, although
! a kpoint typically stashes far fewer MEs (one per fold, e.g. 32).
!-------------------------------------------------------------------------

module vfold_born_batch
  use vfold_mode
  implicit none
  include 'nexternal.inc'
  include 'orders.inc'
  include 'genps.inc'
  include 'born_nhel.inc'
  private :: NEXTERNAL, NINCOMING, NSPLITORDERS, ORDERNAMES, &
       BORN_ORDERS, NLO_ORDERS, QCD_POS, QED_POS, AMP_SPLIT_SIZE, &
       AMP_SPLIT_SIZE_BORN, AMP_SPLIT, AMP_SPLIT_CNT, &
       MAXPROC, NGRAPHS, NCOLOR, MAXFLOW, MAX_BHEL, MAX_BCOL, &
       MAX_PARTICLES, MAX_BRANCH, LMAXCONFIGS
  integer, parameter, public :: VECSIZE_BORN_PROTO = 8192 ! matches MAX_FOLDS_PROTO (step 11g: was 4096)
  ! (see the comment on sde_p_batch below)
  integer, parameter, public :: VECSIZE_SDE_PROTO = 4*VECSIZE_BORN_PROTO
  double precision, public, save :: &
       p_born_batch(0:3, nexternal-1, VECSIZE_BORN_PROTO)
  double precision, public, save :: amp_split_batch(AMP_SPLIT_SIZE, VECSIZE_BORN_PROTO)
  ! Road B step 7h: SBORN's amplitude cache (SAVEAMP/SAVEMOM, born.f) for
  ! each stashed entry (mode C too since step 7i+7j), restored by vfold_retrieve_born. In F0/F1
  ! the other SBORN calls for the same momenta (e.g. in sbornsoft) reuse
  ! the amplitudes computed by compute_born; without this restore they
  ! would find the amplitudes of the last batch entry and recompute.
  ! The private list above includes symbols from genps.inc's own includes.
  double complex, public, save :: &
       saveamp_batch(NGRAPHS, MAX_BHEL, VECSIZE_BORN_PROTO)
  double precision, public, save :: &
       savemom_batch(nexternal-1, 2, VECSIZE_BORN_PROTO)
  double precision, public, save :: f_b_batch(VECSIZE_BORN_PROTO)
  double precision, public, save :: g_batch(VECSIZE_BORN_PROTO)
  double precision, public, save :: wgt_me_born_batch(VECSIZE_BORN_PROTO)
  integer, public, save :: born_batch_count = 0

  ! Road B step 7k: Born memo, keyed on (p_born, g, nfksprocess). All folds
  ! of a kpoint share the same p_born, so a repeated Born evaluation can
  ! skip SBORN entirely (CALCULATEDBORN only skips its HELAS part).
  ! It stores everything SBORN writes that is read downstream: amp_split,
  ! amp_split_cnt, wgt_ME_born, ans_cnt, amp2, jamp2, SAVEAMP, SAVEMOM.
  ! GOODHEL/NTRY are not stored: they are learnt on the first call for
  ! each nfksprocess, which is never a memo hit.
  ! Several slots are needed: the Born is evaluated at several alpha_s
  ! scales (the SDE scale varies per fold), and this Born depends on
  ! alpha_s. About one slot per fold plus one is needed per kpoint
  ! (vfold_born_memo_init), capped at BORN_MEMO_MAX_MB. Too few slots only
  ! cost speed, not correctness.
  integer, parameter, public :: BORN_MEMO_MAX_MB = 200
  integer, public, save :: born_memo_nslot = 0
  integer, public, save :: born_memo_nused = 0
  integer, public, save :: born_memo_next = 1
  double precision, allocatable, public, save :: born_memo_p(:,:,:)
  double precision, allocatable, public, save :: born_memo_g(:)
  integer, allocatable, public, save :: born_memo_nfks(:)
  double precision, allocatable, public, save :: born_memo_amp_split(:,:)
  double complex, allocatable, public, save :: born_memo_amp_split_cnt(:,:,:,:)
  double precision, allocatable, public, save :: born_memo_wgt_me_born(:)
  double complex, allocatable, public, save :: born_memo_ans_cnt(:,:,:)
  double precision, allocatable, public, save :: born_memo_amp2(:,:)
  double precision, allocatable, public, save :: born_memo_jamp2(:,:)
  double complex, allocatable, public, save :: born_memo_saveamp(:,:,:)
  double precision, allocatable, public, save :: born_memo_savemom(:,:,:)
  integer, public, save :: born_memo_found = 0
  ! Road B step 5: cudacpp bridge handle for Born (mode C only). Created
  ! lazily on first use (vfold_evaluate_born_batch); destroyed once per MINT
  ! phase in mint_module.f90's mint() subroutine, mirroring eebb_vfold's
  ! own bridge lifecycle (create once per phase, delete when that phase's
  ! integration loop ends).
  integer*8, public, save :: bridge_born_batch = 0
  ! New counters that were added to investigate VECSIZE_USED
  integer*8, public, save :: vfold_born_batch_sum = 0
  integer*8, public, save :: vfold_born_batch_ncalls = 0

  ! Road B step 3 wiring: born_gathering tells compute_born to stash and
  ! return early instead of evaluating. born_retrieve_idx is walk 2's own
  ! counter, separate from born_batch_count (which must survive from walk
  ! 1 into walk 2 as the "how many were stashed" bound) -- advances in
  ! lockstep with walk 1's stashing IF AND ONLY IF walk 2 reaches
  ! compute_born for exactly the same folds walk 1 did, which is
  ! guaranteed by construction since walk 2 replays the same momenta/
  ! context (via vfold_rng_slots' replay mode) that drove walk 1's decisions.
  logical, public, save :: born_gathering = .false.
  integer, public, save :: born_retrieve_idx = 0

  ! Road B step 7c+7d: batching of the SDE (single diagram enhancement)
  ! Born calls in include_multichannel_enhance (fks_singular.f), which
  ! needs the ratio amp2(selected diagram)/sum(amp2 over diagrams).
  ! The true g is stashed and used: g cancels in the ratio, but a
  ! different g changes the rounding, which the FKS subtractions amplify.
  ! Road B step 7n: modes F and C both evaluate it with vfold_sde_scalar_eval
  ! (scalar SBORN with the Born memo), not with the cudacpp bridge: the
  ! bridge does not fill SBORN's amplitude cache, so the later sborn calls
  ! that rely on it (get_mbar, sbornsoft) would recompute the amplitudes,
  ! which costs as much as the whole scalar SDE evaluation.
  ! VECSIZE_SDE_PROTO > VECSIZE_BORN_PROTO: include_multichannel_enhance
  ! may be called several times per fold (imode=1..4).
  ! As in F0/F1 (step 7f), the SDE Born evaluation time goes to tBorn.
  double precision, public, save :: &
       sde_p_batch(0:3, nexternal-1, VECSIZE_SDE_PROTO)
  integer, public, save :: sde_channelid_batch(VECSIZE_SDE_PROTO)
  double precision, public, save :: sde_g_batch(VECSIZE_SDE_PROTO)
  double precision, public, save :: sde_nomc_mes_batch(VECSIZE_SDE_PROTO)
  double precision, public, save :: sde_mc_mes_batch(VECSIZE_SDE_PROTO)
  integer, public, save :: sde_batch_count = 0
  integer, public, save :: sde_retrieve_idx = 0
  logical, public, save :: sde_gathering = .false.

  ! Road B step 11d (ENV_VFOLD_BATCH_PERVN=1): per-queue copies of the batch
  ! state parked by vfold_born_batch_park when another MINT2 queue is about to
  ! overwrite the batch buffers (see vfold_mint2_park in mint_module.f90), so
  ! that vfold_born_batch_unpark can bring them back instead of regathering.
  ! Only what the retrieve routines read is parked, and only for the entries
  ! not retrieved yet (at the same indices). Allocated on first use.
  integer, parameter :: VFOLD_NPARK = 3 ! one per MINT2 queue (VFOLD_NVN in vfold_rng_slots)
  type vfold_born_parked_t
    integer :: born_count = 0, born_idx = 0, sde_count = 0, sde_idx = 0
    double precision, allocatable :: amp_split(:,:), wgt_me_born(:), f_b(:), g(:)
    double complex, allocatable :: saveamp(:,:,:)
    double precision, allocatable :: savemom(:,:,:)
    double precision, allocatable :: sde_mc_mes(:), sde_nomc_mes(:)
  end type vfold_born_parked_t
  type(vfold_born_parked_t), save :: born_parked(VFOLD_NPARK)

contains

  ! Stash the momenta currently in /pborn/'s p_born, plus f_b and g
  ! (needed later by compute_born's own iamp loop, not by SBORN itself),
  ! for later batched evaluation.
  subroutine vfold_stash_born(f_b, g)
    implicit none
    double precision, intent(in) :: f_b, g
    double precision :: p_born(0:3, nexternal-1)
    common /pborn/ p_born
    integer :: islot
    if (born_batch_count.ge.VECSIZE_BORN_PROTO) then
      write(6,*) 'ERROR vfold_stash_born: batch full, VECSIZE_BORN_PROTO=', &
                 VECSIZE_BORN_PROTO
      stop 1
    endif
    born_batch_count = born_batch_count + 1
    islot = born_batch_count
    p_born_batch(0:3,1:nexternal-1,islot) = p_born(0:3,1:nexternal-1)
    f_b_batch(islot) = f_b
    g_batch(islot) = g
  end subroutine vfold_stash_born

  ! Evaluate once per stashed entry. Mode F: plain Fortran loop calling
  ! SBORN, capturing both of its side-effect outputs (amp_split via
  ! /TO_AMP_SPLIT/, wgt_ME_born via /C_WGT_ME_TREE/) into the batch
  ! arrays. Mode C: one vectorized call to the cudacpp Born bridge instead
  ! (step 5) -- see this file's header comment for the padding/amp_split-
  ! reconstruction details, and for the g-refresh bugfix in the F branch.
  subroutine vfold_evaluate_born_batch
    implicit none
    ! COMMON /timings/ from timing_variables.inc (fixed-form, so it cannot
    ! be included here): same names, types and order as in the .inc file.
    real :: tBorn,tIS,tReal,tCount,tFxFx,tf_nb,tf_all,t_as,tr_s,tr_pdf, &
         t_plot,t_cuts,t_MC_subt,t_isum,t_p_unw,t_write,tOLP,tGenPS, &
         t_ewsud,t_coupl
    common/timings/tBorn,tIS,tReal,tCount,tFxFx,tf_nb,tf_all,t_as, &
         tr_s,tr_pdf,t_plot,t_cuts,t_MC_subt,t_isum,t_p_unw,t_write, &
         tOLP,tGenPS,t_ewsud,t_coupl
    double precision :: p_born(0:3, nexternal-1)
    common /pborn/ p_born
    double precision :: amp_split(AMP_SPLIT_SIZE)
    double complex :: amp_split_cnt(AMP_SPLIT_SIZE,2,NSPLITORDERS)
    common /to_amp_split/ amp_split, amp_split_cnt
    double precision :: wgt_me_born, wgt_me_real
    common /c_wgt_me_tree/ wgt_me_born, wgt_me_real
    double precision :: wgt_c
    logical :: calculatedBorn
    common/ccalculatedBorn/calculatedBorn
    double complex :: saveamp(NGRAPHS, MAX_BHEL)
    common/to_saveamp/saveamp
    double precision :: savemom(nexternal-1, 2)
    common/to_savemom/savemom
    double precision :: g, all_g
    common/strong/ g, all_g
    integer :: i, ipad, j
    logical :: same_as_prev
    integer :: nuniq, uniq_idx(VECSIZE_BORN_PROTO)
    double precision :: uniq_p(0:3, nexternal-1, VECSIZE_BORN_PROTO)
    double precision :: uniq_g(VECSIZE_BORN_PROTO)
    double precision :: uniq_mes(VECSIZE_BORN_PROTO)
    double precision :: mes_bv(VECSIZE_BORN_PROTO)
    double precision :: rnd_bv(VECSIZE_BORN_PROTO)
    integer :: isel_bv(VECSIZE_BORN_PROTO)
    ! The Born MEs are now evaluated here, not in compute_born: add their
    ! cost to tBorn ("Time spent in Born"), once per batch.
    real :: avtiming_t0, avtiming_t1
    ! Chunk buffers for the bridge calls, sized to the maximum chunk size.
    double precision :: chunk_p(0:3,nexternal-1,VECSIZE_BORN_PROTO)
    double precision :: chunk_g(VECSIZE_BORN_PROTO)
    double precision :: chunk_mes(VECSIZE_BORN_PROTO)
    integer :: chunk_start, chunk_end, nthis

    if (born_batch_count.eq.0) return

    vfold_born_batch_sum = vfold_born_batch_sum + born_batch_count
    vfold_born_batch_ncalls = vfold_born_batch_ncalls + 1

    if (vfold_use_cudacpp()) then
      if (bridge_born_batch.eq.0) then
        if (vfold_vecsize_used.gt.VECSIZE_BORN_PROTO) then
          write(6,*) 'ERROR: ENV_VFOLD_VECSIZE_USED=', vfold_vecsize_used, &
                     ' exceeds VECSIZE_BORN_PROTO=', VECSIZE_BORN_PROTO
          stop 1
        endif
        call vfold_tbridge_start ! step 11j
        call FBRIDGE_BORN_CREATE(bridge_born_batch, vfold_vecsize_used, nexternal-1, 4)
        call vfold_tbridge_stop
      endif
      rnd_bv(1:vfold_vecsize_used) = 0d0
      isel_bv(1:vfold_vecsize_used) = 0
      call cpu_time(avtiming_t0)
      ! Road B step 7k: send only the distinct (p_born,g) to the bridge
      ! (duplicates are consecutive), then scatter the results back.
      nuniq = 0
      do i=1,born_batch_count
        same_as_prev = .false.
        if (i.gt.1) then
          same_as_prev = (g_batch(i).eq.g_batch(i-1))
          if (same_as_prev) then
            do j=1,nexternal-1
              if (any(p_born_batch(0:3,j,i).ne.p_born_batch(0:3,j,i-1))) then
                same_as_prev = .false.
                exit
              endif
            enddo
          endif
        endif
        if (.not.same_as_prev) then
          nuniq = nuniq + 1
          uniq_p(0:3,1:nexternal-1,nuniq) = p_born_batch(0:3,1:nexternal-1,i)
          uniq_g(nuniq) = g_batch(i)
        endif
        uniq_idx(i) = nuniq
      enddo
      do chunk_start = 1, nuniq, vfold_vecsize_used
        chunk_end = min(chunk_start + vfold_vecsize_used - 1, nuniq)
        nthis = chunk_end - chunk_start + 1
        chunk_p(0:3,1:nexternal-1,1:nthis) = uniq_p(0:3,1:nexternal-1,chunk_start:chunk_end)
        chunk_g(1:nthis) = uniq_g(chunk_start:chunk_end)
        do ipad = nthis+1, vfold_vecsize_used
          chunk_p(0:3,1:nexternal-1,ipad) = chunk_p(0:3,1:nexternal-1,nthis)
          chunk_g(ipad) = chunk_g(nthis)
        enddo
        call FBRIDGE_BORN_SEQUENCE_NOMULTICHANNEL(bridge_born_batch, &
             chunk_p(:,:,1:vfold_vecsize_used), &
             chunk_g(1:vfold_vecsize_used), rnd_bv(1:vfold_vecsize_used), rnd_bv(1:vfold_vecsize_used), &
             chunk_mes(1:vfold_vecsize_used), isel_bv(1:vfold_vecsize_used), isel_bv(1:vfold_vecsize_used), .false.)
        uniq_mes(chunk_start:chunk_end) = chunk_mes(1:nthis)
      enddo
      do i=1,born_batch_count
        mes_bv(i) = uniq_mes(uniq_idx(i))
      enddo
      call cpu_time(avtiming_t1)
      tBorn = tBorn + (avtiming_t1-avtiming_t0)
      do i=1,born_batch_count
        amp_split_batch(1:AMP_SPLIT_SIZE,i) = 0d0
        amp_split_batch(1,i) = mes_bv(i)
        wgt_me_born_batch(i) = mes_bv(i)
      enddo
      ! Road B step 7i+7j: fill SBORN's amplitude cache for each entry
      ! (restored by vfold_retrieve_born), as mode F gets it from its own
      ! SBORN calls. This needs scalar SBORN calls, but only one per
      ! distinct (p_born,g): all folds of a kpoint share the same p_born,
      ! and the other entries copy the previous one. g must match too,
      ! because SAVEAMP depends on the couplings.
      call cpu_time(avtiming_t0)
      do i=1,born_batch_count
        same_as_prev = .false.
        if (i.gt.1) then
          same_as_prev = (g_batch(i).eq.g_batch(i-1))
          if (same_as_prev) then
            do j=1,nexternal-1
              if (any(p_born_batch(0:3,j,i).ne.p_born_batch(0:3,j,i-1))) then
                same_as_prev = .false.
                exit
              endif
            enddo
          endif
        endif
        if (same_as_prev) then
          saveamp_batch(1:NGRAPHS,1:MAX_BHEL,i) = saveamp_batch(1:NGRAPHS,1:MAX_BHEL,i-1)
          savemom_batch(1:nexternal-1,1:2,i) = savemom_batch(1:nexternal-1,1:2,i-1)
        else
          p_born(0:3,1:nexternal-1) = p_born_batch(0:3,1:nexternal-1,i)
          g = g_batch(i)
          call update_as_param()
          calculatedBorn = .false.
          call sborn(p_born, wgt_c)
          ! Road B step 7l: also store it in the Born memo, as in mode F
          ! (used to re-warm the amplitude cache in the replay walk)
          call vfold_born_memo_store(p_born, g)
          saveamp_batch(1:NGRAPHS,1:MAX_BHEL,i) = saveamp(1:NGRAPHS,1:MAX_BHEL)
          savemom_batch(1:nexternal-1,1:2,i) = savemom(1:nexternal-1,1:2)
        endif
      enddo
      call cpu_time(avtiming_t1)
      tBorn = tBorn + (avtiming_t1-avtiming_t0)
    else
      call cpu_time(avtiming_t0)
      do i=1,born_batch_count
        p_born(0:3,1:nexternal-1) = p_born_batch(0:3,1:nexternal-1,i)
        ! g alone is not enough: SBORN calls HELAS routines using the
        ! derived complex couplings (GC_1, GC_21, ... in coupl.inc's
        ! /couplings/ common), not raw g directly -- see header bugfix note.
        g = g_batch(i)
        call update_as_param()
        ! Road B step 7k: evaluate SBORN only for a new (p_born,g,nfks)
        if (vfold_born_memo_hit(p_born,g)) then
          call vfold_born_memo_restore
        else
          calculatedBorn = .false. ! genuine re-evaluation for a new point
          call sborn(p_born, wgt_c)
          call vfold_born_memo_store(p_born,g)
        endif
        amp_split_batch(1:AMP_SPLIT_SIZE,i) = amp_split(1:AMP_SPLIT_SIZE)
        wgt_me_born_batch(i) = wgt_me_born
        ! Road B step 7h: save SBORN's amplitude cache for this entry
        saveamp_batch(1:NGRAPHS,1:MAX_BHEL,i) = saveamp(1:NGRAPHS,1:MAX_BHEL)
        savemom_batch(1:nexternal-1,1:2,i) = savemom(1:nexternal-1,1:2)
      enddo
      call cpu_time(avtiming_t1)
      tBorn = tBorn + (avtiming_t1-avtiming_t0)
    endif
  end subroutine vfold_evaluate_born_batch

  ! Walk 2: retrieve the next cached entry (in the same order it was
  ! stashed in walk 1) and restore SBORN's two side-effect outputs into
  ! their normal common blocks, plus f_b/g via intent(out) arguments, so
  ! compute_born's iamp loop runs completely unmodified from here on.
  subroutine vfold_retrieve_born(f_b, g)
    implicit none
    double precision, intent(out) :: f_b, g
    double precision :: amp_split(AMP_SPLIT_SIZE)
    double complex :: amp_split_cnt(AMP_SPLIT_SIZE,2,NSPLITORDERS)
    common /to_amp_split/ amp_split, amp_split_cnt
    double precision :: wgt_me_born, wgt_me_real
    common /c_wgt_me_tree/ wgt_me_born, wgt_me_real
    logical :: calculatedBorn
    common/ccalculatedBorn/calculatedBorn
    double complex :: saveamp(NGRAPHS, MAX_BHEL)
    common/to_saveamp/saveamp
    double precision :: savemom(nexternal-1, 2)
    common/to_savemom/savemom
    born_retrieve_idx = born_retrieve_idx + 1
    if (born_retrieve_idx.gt.born_batch_count) then
      write(6,*) 'ERROR vfold_retrieve_born: born_retrieve_idx=', &
                 born_retrieve_idx, ' exceeds born_batch_count=', &
                 born_batch_count
      stop 1
    endif
    amp_split(1:AMP_SPLIT_SIZE) = amp_split_batch(1:AMP_SPLIT_SIZE,born_retrieve_idx)
    wgt_me_born = wgt_me_born_batch(born_retrieve_idx)
    f_b = f_b_batch(born_retrieve_idx)
    g = g_batch(born_retrieve_idx)
    ! Road B step 7h, 7i+7j: restore SBORN's amplitude cache for this entry
    ! (filled by vfold_evaluate_born_batch in both modes F and C).
    ! Step 8o: skip the copy if the cache already holds these momenta.
    if (.not.vfold_born_cache_valid(savemom_batch(1,1,born_retrieve_idx))) then
      saveamp(1:NGRAPHS,1:MAX_BHEL) = saveamp_batch(1:NGRAPHS,1:MAX_BHEL,born_retrieve_idx)
      savemom(1:nexternal-1,1:2) = savemom_batch(1:nexternal-1,1:2,born_retrieve_idx)
    endif
    calculatedBorn = .true.
  end subroutine vfold_retrieve_born

  subroutine vfold_reset_born_batch
    implicit none
    born_batch_count = 0
  end subroutine vfold_reset_born_batch

  subroutine vfold_reset_born_retrieve
    implicit none
    born_retrieve_idx = 0
  end subroutine vfold_reset_born_retrieve

  ! Road B step 7c+7d: stash momenta, diagram number and g (walk 1).
  subroutine vfold_stash_sde(p_born_in, channelid, g_in)
    implicit none
    double precision, intent(in) :: p_born_in(0:3, nexternal-1)
    integer, intent(in) :: channelid
    double precision, intent(in) :: g_in
    integer :: islot
    if (sde_batch_count.ge.VECSIZE_SDE_PROTO) then
      write(6,*) 'ERROR vfold_stash_sde: batch full, VECSIZE_SDE_PROTO=', &
                 VECSIZE_SDE_PROTO
      stop 1
    endif
    sde_batch_count = sde_batch_count + 1
    islot = sde_batch_count
    sde_p_batch(0:3,1:nexternal-1,islot) = p_born_in(0:3,1:nexternal-1)
    sde_channelid_batch(islot) = channelid
    sde_g_batch(islot) = g_in
  end subroutine vfold_stash_sde

  ! Road B step 7c+7d: evaluate numerators and denominators of all
  ! stashed entries with vfold_sde_scalar_eval (modes F and C, step 7n).
  subroutine vfold_evaluate_sde_batch
    implicit none
    real :: tBorn,tIS,tReal,tCount,tFxFx,tf_nb,tf_all,t_as,tr_s,tr_pdf, &
         t_plot,t_cuts,t_MC_subt,t_isum,t_p_unw,t_write,tOLP,tGenPS, &
         t_ewsud,t_coupl
    common/timings/tBorn,tIS,tReal,tCount,tFxFx,tf_nb,tf_all,t_as, &
         tr_s,tr_pdf,t_plot,t_cuts,t_MC_subt,t_isum,t_p_unw,t_write, &
         tOLP,tGenPS,t_ewsud,t_coupl
    integer :: i
    real :: avtiming_t0, avtiming_t1

    if (sde_batch_count.eq.0) return

    call cpu_time(avtiming_t0)
    do i=1,sde_batch_count
      call vfold_sde_scalar_eval(sde_p_batch(0:3,1:nexternal-1,i), &
           sde_channelid_batch(i), sde_g_batch(i), &
           sde_mc_mes_batch(i), sde_nomc_mes_batch(i))
    enddo
    call cpu_time(avtiming_t1)
    tBorn = tBorn + (avtiming_t1-avtiming_t0)
  end subroutine vfold_evaluate_sde_batch

  ! Road B step 7c+7d: retrieve numerator and denominator of the next
  ! stashed entry (walk 2).
  subroutine vfold_retrieve_sde(numerator, denominator)
    implicit none
    double precision, intent(out) :: numerator, denominator
    sde_retrieve_idx = sde_retrieve_idx + 1
    if (sde_retrieve_idx.gt.sde_batch_count) then
      write(6,*) 'ERROR vfold_retrieve_sde: sde_retrieve_idx=', &
                 sde_retrieve_idx, ' exceeds sde_batch_count=', &
                 sde_batch_count
      stop 1
    endif
    numerator = sde_mc_mes_batch(sde_retrieve_idx)
    denominator = sde_nomc_mes_batch(sde_retrieve_idx)
  end subroutine vfold_retrieve_sde

  ! Road B step 7c+7d
  subroutine vfold_reset_sde_batch
    implicit none
    sde_batch_count = 0
  end subroutine vfold_reset_sde_batch

  ! Road B step 7c+7d
  subroutine vfold_reset_sde_retrieve
    implicit none
    sde_retrieve_idx = 0
  end subroutine vfold_reset_sde_retrieve

  ! Road B step 11d: park the Born and SDE batch state of MINT2 queue iq
  ! (the entries not retrieved yet).
  subroutine vfold_born_batch_park(iq)
    implicit none
    integer, intent(in) :: iq
    integer :: i0, i1
    associate(pk => born_parked(iq))
      if (.not.allocated(pk%f_b)) then
        allocate(pk%amp_split(AMP_SPLIT_SIZE,VECSIZE_BORN_PROTO), &
             pk%wgt_me_born(VECSIZE_BORN_PROTO), pk%f_b(VECSIZE_BORN_PROTO), &
             pk%g(VECSIZE_BORN_PROTO), &
             pk%saveamp(NGRAPHS,MAX_BHEL,VECSIZE_BORN_PROTO), &
             pk%savemom(nexternal-1,2,VECSIZE_BORN_PROTO), &
             pk%sde_mc_mes(VECSIZE_SDE_PROTO), pk%sde_nomc_mes(VECSIZE_SDE_PROTO))
      endif
      pk%born_count = born_batch_count
      pk%born_idx = born_retrieve_idx
      i0 = born_retrieve_idx + 1
      i1 = born_batch_count
      if (i1.ge.i0) then
        pk%amp_split(:,i0:i1) = amp_split_batch(:,i0:i1)
        pk%wgt_me_born(i0:i1) = wgt_me_born_batch(i0:i1)
        pk%f_b(i0:i1) = f_b_batch(i0:i1)
        pk%g(i0:i1) = g_batch(i0:i1)
        pk%savemom(:,:,i0:i1) = savemom_batch(:,:,i0:i1)
        pk%saveamp(:,:,i0:i1) = saveamp_batch(:,:,i0:i1)
      endif
      pk%sde_count = sde_batch_count
      pk%sde_idx = sde_retrieve_idx
      i0 = sde_retrieve_idx + 1
      i1 = sde_batch_count
      if (i1.ge.i0) then
        pk%sde_mc_mes(i0:i1) = sde_mc_mes_batch(i0:i1)
        pk%sde_nomc_mes(i0:i1) = sde_nomc_mes_batch(i0:i1)
      endif
    end associate
  end subroutine vfold_born_batch_park

  ! Road B step 11d: bring back what vfold_born_batch_park(iq) parked.
  subroutine vfold_born_batch_unpark(iq)
    implicit none
    integer, intent(in) :: iq
    integer :: i0, i1
    associate(pk => born_parked(iq))
      born_batch_count = pk%born_count
      born_retrieve_idx = pk%born_idx
      i0 = born_retrieve_idx + 1
      i1 = born_batch_count
      if (i1.ge.i0) then
        amp_split_batch(:,i0:i1) = pk%amp_split(:,i0:i1)
        wgt_me_born_batch(i0:i1) = pk%wgt_me_born(i0:i1)
        f_b_batch(i0:i1) = pk%f_b(i0:i1)
        g_batch(i0:i1) = pk%g(i0:i1)
        savemom_batch(:,:,i0:i1) = pk%savemom(:,:,i0:i1)
        saveamp_batch(:,:,i0:i1) = pk%saveamp(:,:,i0:i1)
      endif
      sde_batch_count = pk%sde_count
      sde_retrieve_idx = pk%sde_idx
      i0 = sde_retrieve_idx + 1
      i1 = sde_batch_count
      if (i1.ge.i0) then
        sde_mc_mes_batch(i0:i1) = pk%sde_mc_mes(i0:i1)
        sde_nomc_mes_batch(i0:i1) = pk%sde_nomc_mes(i0:i1)
      endif
    end associate
  end subroutine vfold_born_batch_unpark

  ! Road B step 7k: allocate the Born memo with nkp*(nfold+1) slots, capped at
  ! BORN_MEMO_MAX_MB (nfold = folds per kpoint, nkp = kpoints per batch). Called once from
  ! driver_mintMC.f when the folding is known; if never called, the memo
  ! stays empty and every lookup misses.
  subroutine vfold_born_memo_init(nfold)
    implicit none
    integer, intent(in) :: nfold
    integer :: want, cap, nkp
    integer*8 :: bytes_per_slot
    if (born_memo_nslot.gt.0) return
    ! With kpoint batching nkp kpoints are in flight at once. p_born is shared
    ! across the folds of ONE kpoint only, so the memo needs nkp*(nfold+1)
    ! slots: if smaller, vfold_evaluate_born_batch evicts kpoint 1's entries
    ! before its replay walk looks them up (MINT1, folding 8,4,1, batch 32:
    ! MCsubtraction 0.039 -> 0.231, Counter_terms 0.126 -> 0.315).
    ! nkp is capped as in vfold_kp_batch_size.
    nkp = max(1, min(vfold_kpoint_batch, VECSIZE_BORN_PROTO/max(nfold,1)))
    want = nkp * (max(nfold,1) + 1)
    bytes_per_slot = &
         int(4*(nexternal-1),8)*8 &                       ! p
       + 8_8 + 4_8 &                                      ! g, nfks
       + int(AMP_SPLIT_SIZE,8)*8 &                        ! amp_split
       + int(AMP_SPLIT_SIZE,8)*2*NSPLITORDERS*16 &        ! amp_split_cnt
       + 8_8 &                                            ! wgt_me_born
       + int(2*NSPLITORDERS,8)*16 &                       ! ans_cnt
       + int(NGRAPHS,8)*8 &                               ! amp2
       + int(NCOLOR+1,8)*8 &                              ! jamp2
       + int(NGRAPHS,8)*MAX_BHEL*16 &                     ! saveamp
       + int(nexternal-1,8)*2*8                           ! savemom
    cap = int(max(1_8, (int(BORN_MEMO_MAX_MB,8)*1024*1024)/bytes_per_slot))
    born_memo_nslot = min(want, cap)
    allocate(born_memo_p(0:3, nexternal-1, born_memo_nslot))
    allocate(born_memo_g(born_memo_nslot))
    allocate(born_memo_nfks(born_memo_nslot))
    allocate(born_memo_amp_split(AMP_SPLIT_SIZE, born_memo_nslot))
    allocate(born_memo_amp_split_cnt(AMP_SPLIT_SIZE,2,NSPLITORDERS,born_memo_nslot))
    allocate(born_memo_wgt_me_born(born_memo_nslot))
    allocate(born_memo_ans_cnt(2,NSPLITORDERS,born_memo_nslot))
    allocate(born_memo_amp2(NGRAPHS, born_memo_nslot))
    allocate(born_memo_jamp2(0:NCOLOR, born_memo_nslot))
    allocate(born_memo_saveamp(NGRAPHS, MAX_BHEL, born_memo_nslot))
    allocate(born_memo_savemom(nexternal-1, 2, born_memo_nslot))
    born_memo_nfks(:) = -1
    write(6,*) 'AV: Born memo slots=', born_memo_nslot, &
         ' (folds=', nfold, ', kpoints=', nkp, &
         ', wanted=', want, ', cap=', cap, &
         ', MB=', (bytes_per_slot*born_memo_nslot)/(1024*1024), ')'
    if (born_memo_nslot.lt.want) write(6,*) &
         'AV: WARNING Born memo capped by BORN_MEMO_MAX_MB=', &
         BORN_MEMO_MAX_MB, ' -- slower, still correct'
  end subroutine vfold_born_memo_init

  ! Road B step 7k: does the Born memo hold SBORN's result for exactly
  ! this (p, g, nfksprocess)? All momentum components are compared.
  logical function vfold_born_memo_hit(p, g_in)
    implicit none
    double precision, intent(in) :: p(0:3, nexternal-1), g_in
    integer :: nfksprocess
    common/c_nfksprocess/nfksprocess
    integer :: j, k
    logical :: match
    vfold_born_memo_hit = .false.
    born_memo_found = 0
    if (born_memo_nslot.le.0) return
    do k=1,born_memo_nused
      if (born_memo_g(k).ne.g_in) cycle
      if (born_memo_nfks(k).ne.nfksprocess) cycle
      match = .true.
      do j=1,nexternal-1
        if (any(born_memo_p(0:3,j,k).ne.p(0:3,j))) then
          match = .false.
          exit
        endif
      enddo
      if (match) then
        born_memo_found = k
        vfold_born_memo_hit = .true.
        return
      endif
    enddo
  end function vfold_born_memo_hit

  ! Road B step 7k: store what SBORN just wrote (call right after SBORN).
  subroutine vfold_born_memo_store(p, g_in)
    implicit none
    double precision, intent(in) :: p(0:3, nexternal-1), g_in
    double precision :: amp_split(AMP_SPLIT_SIZE)
    double complex :: amp_split_cnt(AMP_SPLIT_SIZE,2,NSPLITORDERS)
    common /to_amp_split/ amp_split, amp_split_cnt
    double precision :: wgt_me_born, wgt_me_real
    common /c_wgt_me_tree/ wgt_me_born, wgt_me_real
    double complex :: ans_cnt(2, NSPLITORDERS)
    common /c_born_cnt/ ans_cnt
    double precision :: amp2(NGRAPHS), jamp2(0:NCOLOR)
    common /to_amps/ amp2, jamp2
    double complex :: saveamp(NGRAPHS, MAX_BHEL)
    common/to_saveamp/saveamp
    double precision :: savemom(nexternal-1, 2)
    common/to_savemom/savemom
    integer :: nfksprocess
    common/c_nfksprocess/nfksprocess
    integer :: k
    if (born_memo_nslot.le.0) return
    ! Round-robin replacement
    if (born_memo_nused.lt.born_memo_nslot) then
      born_memo_nused = born_memo_nused + 1
      k = born_memo_nused
    else
      k = born_memo_next
      born_memo_next = born_memo_next + 1
      if (born_memo_next.gt.born_memo_nslot) born_memo_next = 1
    endif
    born_memo_p(0:3,1:nexternal-1,k) = p(0:3,1:nexternal-1)
    born_memo_g(k) = g_in
    born_memo_nfks(k) = nfksprocess
    born_memo_amp_split(1:AMP_SPLIT_SIZE,k) = amp_split(1:AMP_SPLIT_SIZE)
    born_memo_amp_split_cnt(:,:,:,k) = amp_split_cnt
    born_memo_wgt_me_born(k) = wgt_me_born
    born_memo_ans_cnt(:,:,k) = ans_cnt
    born_memo_amp2(1:NGRAPHS,k) = amp2(1:NGRAPHS)
    born_memo_jamp2(0:NCOLOR,k) = jamp2(0:NCOLOR)
    born_memo_saveamp(1:NGRAPHS,1:MAX_BHEL,k) = saveamp(1:NGRAPHS,1:MAX_BHEL)
    born_memo_savemom(1:nexternal-1,1:2,k) = savemom(1:nexternal-1,1:2)
  end subroutine vfold_born_memo_store

  ! Road B step 8o: is SBORN's amplitude cache already valid for momenta sm?
  ! Same rule as SBORN (born.f): CALCULATEDBORN set and SAVEMOM unchanged.
  ! Comparing SAVEMOM is ~1000x cheaper than copying SAVEAMP (16 kB here).
  logical function vfold_born_cache_valid(sm)
    implicit none
    double precision, intent(in) :: sm(nexternal-1, 2)
    double precision :: savemom(nexternal-1, 2)
    common/to_savemom/savemom
    logical :: calculatedBorn
    common/ccalculatedBorn/calculatedBorn
    integer :: j
    vfold_born_cache_valid = .false.
    if (.not.calculatedBorn) return
    do j=1,nexternal-1
      if (savemom(j,1).ne.sm(j,1) .or. savemom(j,2).ne.sm(j,2)) return
    enddo
    vfold_born_cache_valid = .true.
  end function vfold_born_cache_valid

  ! Road B step 7k: restore the memo entry found by vfold_born_memo_hit into
  ! the common blocks filled by SBORN.
  subroutine vfold_born_memo_restore
    implicit none
    double precision :: amp_split(AMP_SPLIT_SIZE)
    double complex :: amp_split_cnt(AMP_SPLIT_SIZE,2,NSPLITORDERS)
    common /to_amp_split/ amp_split, amp_split_cnt
    double precision :: wgt_me_born, wgt_me_real
    common /c_wgt_me_tree/ wgt_me_born, wgt_me_real
    double complex :: ans_cnt(2, NSPLITORDERS)
    common /c_born_cnt/ ans_cnt
    double precision :: amp2(NGRAPHS), jamp2(0:NCOLOR)
    common /to_amps/ amp2, jamp2
    double complex :: saveamp(NGRAPHS, MAX_BHEL)
    common/to_saveamp/saveamp
    double precision :: savemom(nexternal-1, 2)
    common/to_savemom/savemom
    logical :: calculatedBorn
    common/ccalculatedBorn/calculatedBorn
    integer :: k
    k = born_memo_found
    amp_split(1:AMP_SPLIT_SIZE) = born_memo_amp_split(1:AMP_SPLIT_SIZE,k)
    amp_split_cnt = born_memo_amp_split_cnt(:,:,:,k)
    wgt_me_born = born_memo_wgt_me_born(k)
    ans_cnt = born_memo_ans_cnt(:,:,k)
    amp2(1:NGRAPHS) = born_memo_amp2(1:NGRAPHS,k)
    jamp2(0:NCOLOR) = born_memo_jamp2(0:NCOLOR,k)
    ! Step 8o: no vfold_born_cache_valid guard here, as it was measured to be
    ! slower (the cache is usually cold when the memo is restored).
    saveamp(1:NGRAPHS,1:MAX_BHEL) = born_memo_saveamp(1:NGRAPHS,1:MAX_BHEL,k)
    savemom(1:nexternal-1,1:2) = born_memo_savemom(1:nexternal-1,1:2,k)
    calculatedBorn = .true.
  end subroutine vfold_born_memo_restore


  ! Road B step 9l: VECSIZE_BORN_PROTO as compiled into this object file
  ! (see vfold_rng_slots_proto)
  integer function vfold_born_batch_proto()
    implicit none
    vfold_born_batch_proto = VECSIZE_BORN_PROTO
  end function vfold_born_batch_proto

end module vfold_born_batch
