! Copyright (C) 2026 CERN.
! Licensed under the GNU Lesser General Public License (version 3 or later).
! Created by A. Valassi (Aug 2026) for project MadVFold (arXiv:2609.23500).
!-------------------------------------------------------------------------
! Road B prototype (eebb_vfalt) step 4. Real batching.
! Batch/cache mechanism for deferring the expensive SMATRIX_REAL evaluation
! inside SREAL (fks_singular.f), mirroring the old two-pass architecture's
! own real_collecting/real_batch_ready design (see eebb_vfold/SubProcesses/
! fks_singular.f's sreal(), the proven reference for this exact problem)
! rather than vfold_born_batch.f90's per-caller stash pattern.
!
! Why this shape, not vfold_born_batch's: compute_born gates itself (stash
! and return early) because every call to compute_born does real, non-
! trivial work. SREAL is different -- three of its four call sites
! (compute_soft_counter_term, compute_collinear_counter_term,
! compute_soft_collinear_counter_term) always pass xi_i_fks=0 or
! y_ij_fks=1 exactly, so they never reach the expensive SMATRIX_REAL
! branch inside sreal() at all -- they always route to the cheap
! sbornsoft/sborncol_isr/sborncol_fsr kernels. Only compute_real_emission
! (generic xi/y) can ever reach the expensive branch. So the gating
! belongs inside sreal() itself, on just that one branch -- not in the
! four calling wrapper subroutines (which is what an earlier version of
! this file tried, and which broke: driver_mintMC.f skips
! compute_soft/collinear/soft_collinear_counter_term and
! compute_MC_subt_term entirely during the gathering walk via
! born_gathering, since their results aren't used there and computing
! them costs real cycles -- but it does NOT skip compute_real_emission,
! which must run in both walks precisely so that sreal()'s own gate
! below gets exercised during gathering).
!
! g (the strong coupling, /strong/ common in coupl.inc) is stashed
! alongside the momenta and restored before each batched SMATRIX_REAL
! call: SMATRIX_REAL reads g from that common block internally rather
! than taking it as an argument, and folds evaluated in the same batch
! can have different g values (set via set_alphaS per fold), so the
! ambient value at evaluate-batch time (whatever the last-processed
! fold during walk 1 left behind) cannot be trusted for earlier entries.
!-------------------------------------------------------------------------
! Road B step 6. Real batching and cudacpp offloading.
! Now vfold_evaluate_real_batch has two implementations, selected by
! vfold_use_cudacpp() (vfold_mode.f90), mirroring vfold_evaluate_born_batch in step 5.
! Mode F keeps calling SMATRIX_REAL in a plain Fortran loop as in step 4.
! Mode C instead calls the cudacpp Real bridge (FBRIDGE_REAL_*, already
! exported by vfold_cudacpp_dispatcher.cpp in step 5) once per kpoint,
! evaluating the full fixed VECSIZE_REAL_PROTO-wide vector every time,
! padding unused slots by repeating the last real entry (same as Born).
! In both modes wgt_ME_real is now explicitly captured/restored (a fix in
! step 6, not just a bridge-mode addition): step 4's Fortran-loop batch
! evaluator called SMATRIX_REAL in a loop without saving wgt_ME_real per
! entry, so by retrieve time it only ever held whatever the LAST loop
! iteration left behind -- harmless for step 4's own bit-identity checks
! (empirically confirmed against F0, so nothing consumed by those checks
! actually depended on it), but the same class of latent gap step 5
! deliberately closed for wgt_ME_born. Closed here the same way, in both
! modes, so nothing is left relying on incidental correctness.
!-------------------------------------------------------------------------

module vfold_real_batch
  use vfold_mode
  implicit none
  include 'nexternal.inc'
  include 'orders.inc'
  private :: NEXTERNAL, NINCOMING, NSPLITORDERS, ORDERNAMES, &
       BORN_ORDERS, NLO_ORDERS, QCD_POS, QED_POS, AMP_SPLIT_SIZE, &
       AMP_SPLIT_SIZE_BORN, AMP_SPLIT, AMP_SPLIT_CNT
  integer, parameter, public :: VECSIZE_REAL_PROTO = 8192 ! matches MAX_FOLDS_PROTO (step 11g: was 4096)

  ! Road B step 4: real_gathering tells sreal()'s hard-point branch to
  ! stash and skip SMATRIX_REAL during walk 1 (mirrors born_gathering).
  logical, public, save :: real_gathering = .false.

  double precision, public, save :: real_momenta_batch(0:3, nexternal, VECSIZE_REAL_PROTO)
  double precision, public, save :: gs_real_batch(VECSIZE_REAL_PROTO)
  double precision, public, save :: real_mes_bv(VECSIZE_REAL_PROTO) ! raw SMATRIX_REAL wgt (pre xi^2*(1-y) damping, pre amp_split scaling)
  ! Step 8a: full amp_split per entry (SMATRIX_REAL may fill several slots).
  ! Mode C keeps the single slot AMP_SPLIT_SIZE_BORN+1 (no per-order bridge output).
  double precision, public, save :: real_amp_split_bv(AMP_SPLIT_SIZE, VECSIZE_REAL_PROTO)
  double precision, public, save :: wgt_me_real_bv(VECSIZE_REAL_PROTO) ! step 6: explicit wgt_ME_real per entry
  ! Step 8a: nFKSprocess per entry (SMATRIX_REAL dispatches on it).
  integer, public, save :: nfks_batch(VECSIZE_REAL_PROTO)
  integer, public, save :: real_batch_count = 0
  integer, public, save :: real_retrieve_idx = 0
  ! Road B step 11d (ENV_VFOLD_BATCH_PERVN=1): per-queue copies of the batch
  ! state parked by vfold_real_batch_park (see vfold_born_batch_park).
  integer, parameter :: VFOLD_NPARK_REAL = 3 ! one per MINT2 queue (VFOLD_NVN in vfold_rng_slots)
  type vfold_real_parked_t
    integer :: count = 0, idx = 0
    double precision, allocatable :: mes(:), wgt_me_real(:), amp_split(:,:)
  end type vfold_real_parked_t
  type(vfold_real_parked_t), save :: real_parked(VFOLD_NPARK_REAL)
  ! Road B step 6: cudacpp bridge handles for Real (mode C only), created
  ! lazily on first use, destroyed once per MINT phase in mint_module.f90.
  ! Step 8c: one handle per Real library real<N> (NFKS_REAL_LIBRARY in
  ! real_me_chooser.f maps nFKSprocess to N).
  integer, parameter, public :: MAX_REAL_LIBS = 9
  integer*8, public, save :: bridge_real_batch(MAX_REAL_LIBS) = 0
  ! Step 8c: memoised FBRIDGE_REAL_LIBRARY_EXISTS (missing libraries fall
  ! back to scalar SMATRIX_REAL).
  logical, save :: real_lib_checked(MAX_REAL_LIBS) = .false.
  logical, save :: real_lib_exists(MAX_REAL_LIBS) = .false.
  ! New counters that were added to investigate VECSIZE_USED
  integer*8, public, save :: vfold_real_batch_sum = 0
  integer*8, public, save :: vfold_real_batch_ncalls = 0

contains

  ! Called from sreal()'s hard-point branch during the gathering walk:
  ! stash momenta + g + nFKSprocess, skip the actual SMATRIX_REAL call.
  subroutine vfold_stash_real(pp, g_in, nfks_in)
    implicit none
    double precision, intent(in) :: pp(0:3,nexternal), g_in
    integer, intent(in) :: nfks_in
    integer :: islot
    if (real_batch_count.ge.VECSIZE_REAL_PROTO) then
      write(6,*) 'ERROR vfold_stash_real: batch full, VECSIZE_REAL_PROTO=', VECSIZE_REAL_PROTO
      stop 1
    endif
    real_batch_count = real_batch_count + 1
    islot = real_batch_count
    real_momenta_batch(0:3,1:nexternal,islot) = pp(0:3,1:nexternal)
    gs_real_batch(islot) = g_in
    nfks_batch(islot) = nfks_in
  end subroutine vfold_stash_real

  ! Step 8c: is libmg5amc_cudacpp_real<n>.so available? (checked once per n)
  logical function vfold_real_lib_available(n)
    implicit none
    integer, intent(in) :: n
    integer :: exists_int
    if (.not.real_lib_checked(n)) then
      call FBRIDGE_REAL_LIBRARY_EXISTS(n, exists_int)
      real_lib_exists(n) = (exists_int.ne.0)
      real_lib_checked(n) = .true.
    endif
    vfold_real_lib_available = real_lib_exists(n)
  end function vfold_real_lib_available

  ! Called once between walk 1 and walk 2 (mint_module.f90's
  ! compute_integrand, alongside vfold_evaluate_born_batch). Mode F: plain
  ! Fortran loop calling SMATRIX_REAL, restoring each entry's own g into
  ! /strong/ before each call (unchanged from step 4). Mode C: one
  ! vectorized call to the cudacpp Real bridge instead (step 6) -- see
  ! this file's header comment for the padding/wgt_ME_real details.
  subroutine vfold_evaluate_real_batch
    implicit none
    ! COMMON /timings/ from timing_variables.inc (see vfold_born_batch.f90).
    real :: tBorn,tIS,tReal,tCount,tFxFx,tf_nb,tf_all,t_as,tr_s,tr_pdf, &
         t_plot,t_cuts,t_MC_subt,t_isum,t_p_unw,t_write,tOLP,tGenPS, &
         t_ewsud,t_coupl
    common/timings/tBorn,tIS,tReal,tCount,tFxFx,tf_nb,tf_all,t_as, &
         tr_s,tr_pdf,t_plot,t_cuts,t_MC_subt,t_isum,t_p_unw,t_write, &
         tOLP,tGenPS,t_ewsud,t_coupl
    double precision :: pp(0:3,nexternal), wgt
    double precision :: g, all_g
    common/strong/ g, all_g
    double precision :: wgt_me_born, wgt_me_real
    common /c_wgt_me_tree/ wgt_me_born, wgt_me_real
    double precision :: amp_split(AMP_SPLIT_SIZE)
    double complex :: amp_split_cnt(AMP_SPLIT_SIZE,2,NSPLITORDERS)
    common /to_amp_split/ amp_split, amp_split_cnt
    integer :: nFKSprocess
    common/c_nFKSprocess/nFKSprocess
    integer :: i, ipad
    double precision :: mes_bv(VECSIZE_REAL_PROTO)
    double precision :: rnd_bv(VECSIZE_REAL_PROTO)
    integer :: isel_bv(VECSIZE_REAL_PROTO)
    ! The Real MEs are now evaluated here, not in sreal: add their cost
    ! to tReal ("Time spent in Reals_evaluation"), once per batch.
    real :: avtiming_t0, avtiming_t1
    ! Chunk buffers for the bridge calls, sized to the maximum chunk size.
    double precision :: chunk_p(0:3,nexternal,VECSIZE_REAL_PROTO)
    double precision :: chunk_g(VECSIZE_REAL_PROTO)
    double precision :: chunk_mes(VECSIZE_REAL_PROTO)
    integer :: chunk_start, chunk_end, nthis
    ! Step 8b+8c: each entry goes to the bridge of its Real library, or to
    ! the scalar SMATRIX_REAL call (as in mode F) if that is not available.
    integer, external :: nfks_real_library
    integer :: nscalar, libnum
    integer :: scalar_idx(VECSIZE_REAL_PROTO)
    integer :: lib_count(MAX_REAL_LIBS)
    integer :: lib_idx(VECSIZE_REAL_PROTO, MAX_REAL_LIBS)

    if (real_batch_count.eq.0) return

    vfold_real_batch_sum = vfold_real_batch_sum + real_batch_count
    vfold_real_batch_ncalls = vfold_real_batch_ncalls + 1

    if (vfold_use_cudacpp()) then
      lib_count(1:MAX_REAL_LIBS) = 0
      nscalar = 0
      do i=1,real_batch_count
        libnum = nfks_real_library(nfks_batch(i))
        if (vfold_real_lib_available(libnum)) then
          lib_count(libnum) = lib_count(libnum) + 1
          lib_idx(lib_count(libnum), libnum) = i
        else
          nscalar = nscalar + 1
          scalar_idx(nscalar) = i
        endif
      enddo
      do libnum=1,MAX_REAL_LIBS
        if (lib_count(libnum).eq.0) cycle
        if (bridge_real_batch(libnum).eq.0) then
          if (vfold_vecsize_used.gt.VECSIZE_REAL_PROTO) then
            write(6,*) 'ERROR: ENV_VFOLD_VECSIZE_USED=', vfold_vecsize_used, &
                       ' exceeds VECSIZE_REAL_PROTO=', VECSIZE_REAL_PROTO
            stop 1
          endif
          call vfold_tbridge_start ! step 11j
          call FBRIDGE_REAL_CREATE(libnum, bridge_real_batch(libnum), vfold_vecsize_used, nexternal, 4)
          call vfold_tbridge_stop
        endif
        rnd_bv(1:vfold_vecsize_used) = 0d0
        isel_bv(1:vfold_vecsize_used) = 0
        call cpu_time(avtiming_t0)
        do chunk_start = 1, lib_count(libnum), vfold_vecsize_used
          chunk_end = min(chunk_start + vfold_vecsize_used - 1, lib_count(libnum))
          nthis = chunk_end - chunk_start + 1
          do i=1,nthis
            chunk_p(0:3,1:nexternal,i) = real_momenta_batch(0:3,1:nexternal,lib_idx(chunk_start+i-1,libnum))
            chunk_g(i) = gs_real_batch(lib_idx(chunk_start+i-1,libnum))
          enddo
          do ipad = nthis+1, vfold_vecsize_used
            chunk_p(0:3,1:nexternal,ipad) = chunk_p(0:3,1:nexternal,nthis)
            chunk_g(ipad) = chunk_g(nthis)
          enddo
          call FBRIDGE_REAL_SEQUENCE_NOMULTICHANNEL(libnum, bridge_real_batch(libnum), &
               chunk_p(:,:,1:vfold_vecsize_used), &
               chunk_g(1:vfold_vecsize_used), rnd_bv(1:vfold_vecsize_used), rnd_bv(1:vfold_vecsize_used), &
               chunk_mes(1:vfold_vecsize_used), isel_bv(1:vfold_vecsize_used), isel_bv(1:vfold_vecsize_used), .false.)
          mes_bv(chunk_start:chunk_end) = chunk_mes(1:nthis)
        enddo
        call cpu_time(avtiming_t1)
        tReal = tReal + (avtiming_t1-avtiming_t0)
        do i=1,lib_count(libnum)
          real_mes_bv(lib_idx(i,libnum)) = mes_bv(i)
          wgt_me_real_bv(lib_idx(i,libnum)) = mes_bv(i)
          ! Step 8a: mode C keeps the single-slot amp_split
          real_amp_split_bv(1:AMP_SPLIT_SIZE,lib_idx(i,libnum)) = 0d0
          real_amp_split_bv(AMP_SPLIT_SIZE_BORN+1,lib_idx(i,libnum)) = mes_bv(i)
        enddo
      enddo
      if (nscalar.gt.0) then
        call cpu_time(avtiming_t0)
        do i=1,nscalar
          pp(0:3,1:nexternal) = real_momenta_batch(0:3,1:nexternal,scalar_idx(i))
          g = gs_real_batch(scalar_idx(i))
          call update_as_param()
          nFKSprocess = nfks_batch(scalar_idx(i))
          call smatrix_real(pp,wgt)
          real_mes_bv(scalar_idx(i)) = wgt
          wgt_me_real_bv(scalar_idx(i)) = wgt_me_real
          real_amp_split_bv(1:AMP_SPLIT_SIZE,scalar_idx(i)) = amp_split(1:AMP_SPLIT_SIZE)
        enddo
        call cpu_time(avtiming_t1)
        tReal = tReal + (avtiming_t1-avtiming_t0)
      endif
    else
      call cpu_time(avtiming_t0)
      do i=1,real_batch_count
        pp(0:3,1:nexternal) = real_momenta_batch(0:3,1:nexternal,i)
        g = gs_real_batch(i)
        ! g alone is not enough: SMATRIX_REAL calls HELAS routines using
        ! the derived complex couplings (GC_1, GC_21, ... in coupl.inc's
        ! /couplings/ common), not raw g directly. set_alphaS's own
        ! set_ren_scale calls update_as_param() right after setting g for
        ! exactly this reason -- do the same here for each stashed g.
        call update_as_param()
        ! Step 8a: restore this entry's nFKSprocess
        nFKSprocess = nfks_batch(i)
        call smatrix_real(pp,wgt)
        real_mes_bv(i) = wgt
        wgt_me_real_bv(i) = wgt_me_real
        ! Step 8a: keep the full amp_split filled by SMATRIX_REAL
        real_amp_split_bv(1:AMP_SPLIT_SIZE,i) = amp_split(1:AMP_SPLIT_SIZE)
      enddo
      call cpu_time(avtiming_t1)
      tReal = tReal + (avtiming_t1-avtiming_t0)
    endif
  end subroutine vfold_evaluate_real_batch

  ! Called from sreal()'s hard-point branch during the replay walk:
  ! return the next batch-evaluated raw wgt and amp_split (step 8a), in
  ! the same order it was stashed in walk 1, instead of calling
  ! SMATRIX_REAL again, and restore wgt_ME_real directly into its common
  ! block (step 6, mirrors how vfold_retrieve_born restores wgt_ME_born).
  subroutine vfold_retrieve_real(amp_split_out, wgt)
    implicit none
    double precision, intent(out) :: amp_split_out(AMP_SPLIT_SIZE)
    double precision, intent(out) :: wgt
    double precision :: wgt_me_born, wgt_me_real
    common /c_wgt_me_tree/ wgt_me_born, wgt_me_real
    real_retrieve_idx = real_retrieve_idx + 1
    if (real_retrieve_idx.gt.real_batch_count) then
      write(6,*) 'ERROR vfold_retrieve_real: real_retrieve_idx=', real_retrieve_idx, &
                 ' exceeds real_batch_count=', real_batch_count
      stop 1
    endif
    wgt = real_mes_bv(real_retrieve_idx)
    wgt_me_real = wgt_me_real_bv(real_retrieve_idx)
    amp_split_out(1:AMP_SPLIT_SIZE) = real_amp_split_bv(1:AMP_SPLIT_SIZE,real_retrieve_idx)
  end subroutine vfold_retrieve_real

  subroutine vfold_reset_real_batch
    implicit none
    real_batch_count = 0
  end subroutine vfold_reset_real_batch

  subroutine vfold_reset_real_retrieve
    implicit none
    real_retrieve_idx = 0
  end subroutine vfold_reset_real_retrieve

  ! Road B step 11d: park the Real batch state of MINT2 queue iq (the
  ! entries not retrieved yet, at the same indices).
  subroutine vfold_real_batch_park(iq)
    implicit none
    integer, intent(in) :: iq
    integer :: i0, i1
    associate(pk => real_parked(iq))
      if (.not.allocated(pk%mes)) allocate(pk%mes(VECSIZE_REAL_PROTO), &
           pk%wgt_me_real(VECSIZE_REAL_PROTO), pk%amp_split(AMP_SPLIT_SIZE,VECSIZE_REAL_PROTO))
      pk%count = real_batch_count
      pk%idx = real_retrieve_idx
      i0 = real_retrieve_idx + 1
      i1 = real_batch_count
      if (i1.ge.i0) then
        pk%mes(i0:i1) = real_mes_bv(i0:i1)
        pk%wgt_me_real(i0:i1) = wgt_me_real_bv(i0:i1)
        pk%amp_split(:,i0:i1) = real_amp_split_bv(:,i0:i1)
      endif
    end associate
  end subroutine vfold_real_batch_park

  ! Road B step 11d: bring back what vfold_real_batch_park(iq) parked.
  subroutine vfold_real_batch_unpark(iq)
    implicit none
    integer, intent(in) :: iq
    integer :: i0, i1
    associate(pk => real_parked(iq))
      real_batch_count = pk%count
      real_retrieve_idx = pk%idx
      i0 = real_retrieve_idx + 1
      i1 = real_batch_count
      if (i1.ge.i0) then
        real_mes_bv(i0:i1) = pk%mes(i0:i1)
        wgt_me_real_bv(i0:i1) = pk%wgt_me_real(i0:i1)
        real_amp_split_bv(:,i0:i1) = pk%amp_split(:,i0:i1)
      endif
    end associate
  end subroutine vfold_real_batch_unpark


  ! Road B step 9l: VECSIZE_REAL_PROTO as compiled into this object file
  ! (see vfold_rng_slots_proto)
  integer function vfold_real_batch_proto()
    implicit none
    vfold_real_batch_proto = VECSIZE_REAL_PROTO
  end function vfold_real_batch_proto
end module vfold_real_batch
