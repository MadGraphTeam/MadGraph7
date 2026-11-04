! Copyright (C) 2026 CERN.
! Licensed under the GNU Lesser General Public License (version 3 or later).
! Created by A. Valassi (Aug 2026) for project MadVFold (arXiv:2609.23500).
!
! Road B prototype (eebb_vfalt): step 3 Born batching standalone test.
! Self-contained correctness test for vfold_born_batch's stash/evaluate
! mechanism -- does NOT touch the real event loop or add_wgt at all.
! Opt-in only via ENV_VFOLD_TEST_BORN_BATCH; when it runs it STOPs the program
! early, no physics run happens.
!
! What it checks: build a handful of distinct hand-built on-shell 2->2
! momenta configurations, stash each one (vfold_stash_born) along with a
! distinct (f_b,g) pair, evaluate the whole batch in one loop
! (vfold_evaluate_born_batch), then retrieve each one in order
! (vfold_retrieve_born) and compare against calling SBORN directly on the
! same momenta one at a time. If stashing/retrieval correctly round-trips
! everything to the right slot, in the right order, these must match
! exactly (same code, same inputs, bit-identical outputs).

subroutine vfold_test_born_batch
  use vfold_born_batch
  implicit none
  include 'nexternal.inc'
  include 'orders.inc'
  character(len=32) :: getenvvalue
  integer :: iconf, i, nfail
  double precision :: theta, ecm, mb, pmag
  double precision :: p_born(0:3, nexternal-1)
  common /pborn/ p_born
  double precision :: wgt_c
  logical :: calculatedBorn
  common/ccalculatedBorn/calculatedBorn
  double precision :: amp_split_direct(AMP_SPLIT_SIZE,10)
  double precision :: f_b_conf(10), g_conf(10)
  double precision :: f_b_out, g_out
  double precision :: pi
  parameter (pi=3.14159265358979d0)
  integer :: nfksprocess
  common/c_nfksprocess/nfksprocess

  call getenv('ENV_VFOLD_TEST_BORN_BATCH', getenvvalue)
  if (getenvvalue.eq.'') return

  write(6,*) 'AV TEST: starting vfold_born_batch stash/evaluate/retrieve test'
  call vfold_reset_born_batch
  call vfold_reset_born_retrieve
  ! SBORN_SPLITORDERS indexes IDEN_VALUES/GOODHEL/NTRY by NFKSPROCESS,
  ! which is normally set by the surrounding sigintF machinery -- this
  ! test runs standalone, before any of that, so it must be set by hand.
  nfksprocess = 1

  ! Build 10 distinct on-shell 2->2 momenta configurations (different
  ! scattering angle each time), each with its own fake (f_b,g) pair, and
  ! for each: (a) compute SBORN directly and remember its amp_split,
  ! (b) stash it into the batch.
  ! NOTE: the outgoing b/bbar are massive (mdl_MB ~ 4.7 GeV) -- pmag must
  ! be sqrt(ecm**2-mb**2), NOT ecm itself, or the momenta are not truly
  ! on-shell (this exact mistake was already flagged once before, in the
  ! vfold_benchmark.f microbenchmark work, where it silently produced a NaN
  ! wgt_c instead of a real answer -- here it silently produced an
  ! all-zero amp_split instead, an equally vacuous "PASS" until checked).
  ecm = 500d0
  mb = 4.7d0
  pmag = sqrt(ecm**2 - mb**2)
  do iconf=1,10
    theta = iconf*pi/11d0 ! 10 distinct angles, none at 0 or pi
    p_born(0,1)=ecm; p_born(1,1)=0d0; p_born(2,1)=0d0; p_born(3,1)=ecm
    p_born(0,2)=ecm; p_born(1,2)=0d0; p_born(2,2)=0d0; p_born(3,2)=-ecm
    p_born(0,3)=ecm
    p_born(1,3)=pmag*sin(theta)
    p_born(2,3)=0d0
    p_born(3,3)=pmag*cos(theta)
    p_born(0,4)=ecm
    p_born(1,4)=-pmag*sin(theta)
    p_born(2,4)=0d0
    p_born(3,4)=-pmag*cos(theta)
    calculatedBorn = .false. ! force genuine re-evaluation, not a cache hit
    call sborn(p_born, wgt_c)
    amp_split_direct(1:AMP_SPLIT_SIZE,iconf) = amp_split_local()
    f_b_conf(iconf) = 1.1d0 + 0.01d0*iconf ! distinct fake f_b per config
    g_conf(iconf) = 1.2d0 + 0.01d0*iconf   ! distinct fake g per config
    call vfold_stash_born(f_b_conf(iconf), g_conf(iconf))
    write(6,*) 'AV TEST: conf', iconf, 'direct amp_split=', &
               amp_split_direct(1:AMP_SPLIT_SIZE,iconf)
  enddo

  call vfold_evaluate_born_batch

  nfail = 0
  do iconf=1,10
    call vfold_retrieve_born(f_b_out, g_out)
    if (f_b_out.ne.f_b_conf(iconf) .or. g_out.ne.g_conf(iconf)) then
      write(6,*) 'AV TEST: MISMATCH conf', iconf, 'f_b/g not retrieved', &
                 ' in the right order: got', f_b_out, g_out, &
                 ' expected', f_b_conf(iconf), g_conf(iconf)
      nfail = nfail + 1
    endif
    do i=1,AMP_SPLIT_SIZE
      if (amp_split_local_i(i).ne.amp_split_direct(i,iconf)) then
        write(6,*) 'AV TEST: MISMATCH conf', iconf, 'i=', i, &
                   'direct=', amp_split_direct(i,iconf), &
                   'batch=', amp_split_local_i(i)
        nfail = nfail + 1
      endif
    enddo
  enddo

  if (nfail.eq.0) then
    write(6,*) 'AV TEST: PASS -- all 10 configs, amp_split AND f_b/g', &
               ' retrieved in the right order, matching direct evaluation'
  else
    write(6,*) 'AV TEST: FAIL --', nfail, 'mismatches found'
  endif

  write(6,*) 'AV TEST: done, stopping (no physics run).'
  stop 0

contains

  ! Small helpers so the main body above doesn't need its own /TO_AMP_SPLIT/
  ! common block declaration (vfold_retrieve_born already puts the result
  ! there for us; these just read it back conveniently).
  function amp_split_local() result(a)
    implicit none
    double precision :: a(AMP_SPLIT_SIZE)
    double precision :: amp_split(AMP_SPLIT_SIZE)
    double complex :: amp_split_cnt(AMP_SPLIT_SIZE,2,NSPLITORDERS)
    common /to_amp_split/ amp_split, amp_split_cnt
    a = amp_split
  end function amp_split_local

  function amp_split_local_i(i) result(a)
    implicit none
    integer, intent(in) :: i
    double precision :: a
    double precision :: amp_split(AMP_SPLIT_SIZE)
    double complex :: amp_split_cnt(AMP_SPLIT_SIZE,2,NSPLITORDERS)
    common /to_amp_split/ amp_split, amp_split_cnt
    a = amp_split(i)
  end function amp_split_local_i

end subroutine vfold_test_born_batch
