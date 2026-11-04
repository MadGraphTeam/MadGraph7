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
! MadVFold codegen step 12h: only the scaffolding is installed here, so
! F1, F2, F and C are not yet implemented: all five modes run identical,
! unmodified code paths, so F0/F1/F2/F/C must all produce bit-identical
! results. Later steps make F1, F2, F and C actually diverge from F0.
module vfold_mode
  implicit none
  character(len=2), public, save :: vfold_me_mode = 'C '
contains

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
end module vfold_mode
