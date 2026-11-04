! Copyright (C) 2026 CERN.
! Licensed under the GNU Lesser General Public License (version 3 or later).
! Created by A. Valassi (Aug 2026) for project MadVFold (arXiv:2609.23500).

! Road B prototype (eebb_vfalt) step 1: F0/F1/F/C mode scaffolding.
! Holds the ENV_VFOLD_ME_MODE flag shared across the process. Five modes:
!   F0 = untouched baseline (never touches the RNG-slot machinery added in
!        later steps; kept as a permanent statistical cross-check)
!   F1 = same single-pass control flow as F0 (no deferred Born/Real
!        evaluation), but random numbers are drawn through the RNG-slot
!        machinery like F: isolates "different control flow" from
!        "different random numbers" in F0-vs-F comparisons (F1 and F
!        are expected to give bit-identical results)
!   F2 = same as F1, plus deduplication of the Born evaluations via the
!        Born memo (all folds of a kpoint share the same p_born)
!   F  = new single-pass Fortran-only implementation (RNG slots + deferred
!        Born/Real evaluation via a plain Fortran loop, no cudacpp bridge)
!   C  = same as F, but Born/Real evaluation is offloaded to the cudacpp
!        bridge instead of a plain Fortran loop
! MadVFold codegen status: F1 is implemented since step 12j (RNG slots)
! and F since step 12k (batched Born/Real MEs over the folds of a kpoint).
! F2 still runs as F1 (the Born memo is only used by F), and C runs as F
! until the cudacpp bridges are added. F1, F2, F and C must all produce
! bit-identical results, different from those of F0.
module vfold_mode
  implicit none
  character(len=2), public, save :: vfold_me_mode = 'C '

  ! Road B step 11q: time spent in the MC@NLO-Delta extra work of
  ! complete_xmcsubt (Pythia8 stopping scales and dead zones, then the Delta
  ! Sudakov factors), printed as 'Time spent in Delta_Pythia', subtracted
  ! from MCsubtraction (compute_MC_subt_term) and from Other_tasks.
  ! It is 0 if mcatnlo_delta is not set.
  real, public, save :: vfold_tdelta = 0
  real, private, save :: vfold_tdelta_t0 = 0
contains

  ! Road B step 11q: time a complete_xmcsubt call into vfold_tdelta.
  subroutine vfold_tdelta_start
    implicit none
    call cpu_time(vfold_tdelta_t0)
  end subroutine vfold_tdelta_start

  subroutine vfold_tdelta_stop
    implicit none
    real :: t1
    call cpu_time(t1)
    vfold_tdelta = vfold_tdelta + (t1-vfold_tdelta_t0)
  end subroutine vfold_tdelta_stop

  subroutine setup_av_me_mode
    implicit none
    character(len=10) :: getenvvalue
    call getenv('ENV_VFOLD_ME_MODE', getenvvalue)
    if (getenvvalue.eq.'F0') then
      vfold_me_mode = 'F0'
    elseif (getenvvalue.eq.'F1') then
      vfold_me_mode = 'F1'
    elseif (getenvvalue.eq.'F2') then
      vfold_me_mode = 'F2'
    elseif (getenvvalue.eq.'F') then
      vfold_me_mode = 'F '
    elseif (getenvvalue.eq.'C') then
      vfold_me_mode = 'C '
    else
      write(6,*) 'ERROR: ENV_VFOLD_ME_MODE must be F0, F1, F2, F or C, got: ', &
                 trim(getenvvalue)
      stop 1
    endif
    write(6,*) 'AV: ENV_VFOLD_ME_MODE=', vfold_me_mode
  end subroutine setup_av_me_mode

  ! Mode properties: later steps should test these instead of comparing
  ! vfold_me_mode against explicit lists of mode strings, so that a new mode
  ! only needs to be added here.

  ! Draw random numbers through the RNG-slot machinery (F1, F2, F, C)
  ! instead of bare ran2() calls (F0)?
  logical function vfold_use_rng_slots()
    implicit none
    vfold_use_rng_slots = (vfold_me_mode.ne.'F0')
  end function vfold_use_rng_slots

  ! Use the batch API, i.e. deferred Born/Real evaluation of all folds of
  ! each kpoint in a batch (F, C), instead of the original immediate
  ! evaluation one fold at a time (F0, F1, F2)?
  logical function vfold_use_batch_api()
    implicit none
    vfold_use_batch_api = (vfold_me_mode.eq.'F ' .or. vfold_me_mode.eq.'C ')
  end function vfold_use_batch_api

  ! Evaluate the batched Born/Real MEs through the cudacpp bridge (C),
  ! instead of a plain Fortran loop over SBORN/SMATRIX_REAL (F)?
  ! Only meaningful when vfold_use_batch_api() is true.
  logical function vfold_use_cudacpp()
    implicit none
    vfold_use_cudacpp = (vfold_me_mode.eq.'C ')
  end function vfold_use_cudacpp

  ! Deduplicate the direct (unbatched) Born evaluations with the Born
  ! memo (F2)? The batched evaluations of F and C always use the memo;
  ! F0 and F1 never do (they are the reference points).
  logical function vfold_use_born_memo()
    implicit none
    vfold_use_born_memo = (vfold_me_mode.eq.'F2')
  end function vfold_use_born_memo

  ! Road B step 11b: keep the original code (F0, F1: the reference points)
  ! also for the changes that are not specific to the new modes, such as
  ! the faster timers of step 11a (ENV_VFOLD_CPUTIME<0, see vfold_cpu_time.c)
  logical function vfold_is_reference_mode()
    implicit none
    vfold_is_reference_mode = .not. (vfold_use_born_memo() .or. vfold_use_batch_api())
  end function vfold_is_reference_mode

  ! Road B step 11b: vfold_cpu_time.c chooses its implementation at the first
  ! cpu_time call, before setup_av_me_mode, reading ENV_VFOLD_ME_MODE itself
  ! for ENV_VFOLD_CPUTIME<0: check that it agrees with vfold_is_reference_mode().
  subroutine check_av_cpu_time
    implicit none
    integer, external :: vfold_cpu_time_mode
    integer :: mode, setting
    mode = vfold_cpu_time_mode(setting)
    if (setting.lt.0) then
      if (vfold_is_reference_mode() .neqv. (mode.eq.0)) then
        write(6,*) 'ERROR: vfold_cpu_time mode=', mode, ' for ENV_VFOLD_CPUTIME=', &
             setting, ' in ENV_VFOLD_ME_MODE=', vfold_me_mode
        stop 1
      endif
    endif
  end subroutine check_av_cpu_time
end module vfold_mode
