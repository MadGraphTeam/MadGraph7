"""Actual producer bank packing, current support and history lifecycle guards."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_born_spreading import mint_support, TEMPLATE


PROGRAM = r'''
program exercise_replay
use mint_module
use,intrinsic :: ieee_arithmetic,only:ieee_value,ieee_quiet_nan
implicit none
character(32) :: action
double precision :: b(2,16,2),r(2,16,2,2),pb(3,16,3),pr(3,16,2,3)
double precision :: saved_b(2,16,2),saved_r(2,16,2,2),saved_pb(3,16,3),saved_pr(3,16,2,3)
double precision,allocatable :: fb(:,:,:),fr(:,:,:,:)
integer :: n,s,j,fv,test_orbit,selected1,selected2,nseed,pdgs(2,5)
integer :: current(16,2),map1(16,2),map2(16,2),q(2,2)
integer,allocatable :: seed_a(:),seed_b(:)
double precision :: before1,after1,before2,after2
type(unified_fit_attempt) :: attempt1(3),attempt2(3)
call get_command_argument(1,action)
do test_orbit=1,2
 do n=1,16
  do fv=1,2
   b(fv,n,test_orbit)=dble(n+3*test_orbit-7*fv)/8d0
   do s=1,2
    r(fv,n,s,test_orbit)=dble(3*n-5*s+2*test_orbit+fv)/16d0
   enddo
  enddo
 enddo
enddo
do test_orbit=1,3
 do n=1,16
  do fv=1,3
   pb(fv,n,test_orbit)=dble(2*n-3*test_orbit+fv)/16d0
   do s=1,2
    pr(fv,n,s,test_orbit)=dble(n+7*s-5*test_orbit-fv)/32d0
   enddo
  enddo
 enddo
enddo
saved_b=b;saved_r=r;saved_pb=pb;saved_pr=pr
do s=1,2
 current(:,s)=[(n,n=1,16)]
enddo
q=0
if(trim(action).eq.'normal')then
 call sector_learning_epoch_replay_bank(b,r,pb,pr,fb,fr)
 if(any(shape(fb).ne.[3,16,5]).or.any(shape(fr).ne.[3,16,2,5]))stop 501
 if(any(fb(1:2,:,1:2).ne.b).or.any(fr(1:2,:,:,1:2).ne.r))stop 502
 if(any(fb(3,:,1:2).ne.0d0).or.any(fr(3,:,:,1:2).ne.0d0))stop 503
 if(any(abs(fb(:,:,3:)-(.5d0*2d0/3d0)*pb).gt.1d-15))stop 504
 if(any(abs(fr(:,:,:,3:)-(.5d0*2d0/3d0)*pr).gt.1d-15))stop 505
 if(any(b.ne.saved_b).or.any(r.ne.saved_r).or.any(pb.ne.saved_pb).or.any(pr.ne.saved_pr))stop 506
 print *,'PASS canonical padding, current-first scale and immutable raw banks'
elseif(trim(action).eq.'empty_previous')then
 call sector_learning_epoch_replay_bank(b,r,pb(:,:,1:0),pr(:,:,:,1:0),fb,fr)
 if(any(shape(fb).ne.shape(b)).or.any(shape(fr).ne.shape(r)))stop 507
 if(any(fb.ne.b).or.any(fr.ne.r))stop 508
 call sector_learning_unified_fit(b,r,current,2,2,q,map1,before1,after1,selected1,attempt1)
 call sector_learning_unified_fit(fb,fr,current,2,2,q,map2,before2,after2,selected2,attempt2)
 if(any(map1.ne.map2).or.before1.ne.before2.or.after1.ne.after2.or.selected1.ne.selected2)stop 509
 do j=1,3
  if(attempt1(j)%status.ne.attempt2(j)%status.or.attempt1(j)%reuse.ne.attempt2(j)%reuse.or. &
       attempt1(j)%loss_start.ne.attempt2(j)%loss_start.or.attempt1(j)%loss_phase.ne.attempt2(j)%loss_phase.or. &
       attempt1(j)%loss_final.ne.attempt2(j)%loss_final)stop 510
  if(allocated(attempt1(j)%phase_trace))then
   if(any(attempt1(j)%phase_trace.ne.attempt2(j)%phase_trace))stop 511
  endif
  if(allocated(attempt1(j)%tile_trace))then
   if(any(attempt1(j)%tile_trace.ne.attempt2(j)%tile_trace))stop 512
  endif
 enddo
 print *,'PASS exact fit/maps/loss/trace fallback for empty previous bank'
elseif(trim(action).eq.'empty_current')then
 call sector_learning_epoch_replay_bank(b(:,:,1:0),r(:,:,:,1:0),pb,pr,fb,fr)
 if(any(shape(fb).ne.[2,16,0]).or.any(shape(fr).ne.[2,16,2,0]))stop 513
 print *,'PASS empty current bank never adds previous observations'
elseif(trim(action).eq.'nan')then
 b(1,1,1)=ieee_value(0d0,ieee_quiet_nan)
 call sector_learning_epoch_replay_bank(b,r,pb,pr,fb,fr)
 stop 514
elseif(trim(action).eq.'badshape')then
 call sector_learning_epoch_replay_bank(b,r,pb,pr(:,:,1:1,:),fb,fr)
 stop 515
else
 call DefaultFKSParam
 sector_learning_active=.true.;sector_learning_archive=.true.;FKSSectorLearningBudget=.35d0
 fixed_order=.false.;born_spread_active=.false.;sl_ready=.false.;imode=0
 ndim=4;nchans=1;ichan=1;nint_used=4;ifold=1
 do j=1,4
  xgrid(0:4,j,1)=[(dble(n)/4d0,n=0,4)]
 enddo
 xacc=0d0;nhits=0
 pdgs=21
 call sector_learning_register_layout(1,[1,2],[1,2],pdgs)
 call sl_unified_ensure()
 sl_entries(1)%ntrain=64;sl_entries(1)%nflavours=2
 allocate(sl_entries(1)%b(2,16,64),sl_entries(1)%r(2,16,2,64))
 sl_entries(1)%b=-2d0;sl_entries(1)%r=0d0
 online_epoch=1
 call sl_epoch_replay_begin_iteration()
 if(epoch_replay_previous(1)%nt.ne.64.or.epoch_replay_previous(1)%nf.ne.2)stop 516
 f=29d0;vtot=13d0;etot=17d0;HwU_values=19d0;ans=23d0;unc=31d0
 rand=.13d0;icell=2;ncell=4;sl_current=1
 online_ordinary_calls=64;online_probes=1;online_reused_nodes=16
 if(trim(action).eq.'runtime_empty')then
  call sl_epoch_replay_fit_epoch()
  if(online_slots(1)%gx.ne.1.or.online_slots(1)%gy.ne.1.or.unified_slots(1)%fit_status.ne.0)stop 517
  if(any(sl_entries(1)%permutations.ne.current).or.any(xacc.ne.0d0))stop 518
  print *,'PASS positive previous bank cannot fit/promote an empty current epoch'
  stop 0
 endif
 if(trim(action).eq.'empty_intervening')then
  ! Real empty reset replaces history; it must not search backward to64.
  call sl_epoch_replay_begin_iteration()
  if(epoch_replay_previous(1)%nt.ne.0)stop 519
 endif
 sl_entries(1)%ntrain=1;sl_entries(1)%nflavours=1
 allocate(sl_entries(1)%b(1,16,1),sl_entries(1)%r(1,16,2,1))
 sl_entries(1)%b=-1d0;sl_entries(1)%r=0d0
 allocate(online_slots(1)%cells(4,16,1),online_slots(1)%channels(1),online_slots(1)%sample_ids(1))
 online_slots(1)%cells=1;online_slots(1)%channels=1;online_slots(1)%sample_ids=64
 call random_seed(size=nseed);allocate(seed_a(nseed),seed_b(nseed));call random_seed(get=seed_a)
 call sl_epoch_replay_fit_epoch()
 call random_seed(get=seed_b)
 if(any(seed_a.ne.seed_b))stop 520
 if(online_slots(1)%gx.ne.1.or.online_slots(1)%gy.ne.1.or.unified_slots(1)%fit_status.ne.1)stop 521
 if(online_slots(1)%total_samples.ne.1.or.sl_entries(1)%ntrain.ne.1.or.sl_entries(1)%nflavours.ne.1)stop 522
 if(any(shape(sl_entries(1)%b).ne.[1,16,1]).or.any(shape(sl_entries(1)%r).ne.[1,16,2,1]))stop 523
 if(any(sl_entries(1)%b.ne.-1d0).or.any(sl_entries(1)%r.ne.0d0))stop 524
 if(any(f.ne.29d0).or.any(vtot.ne.13d0).or.any(etot.ne.17d0).or.any(HwU_values.ne.19d0))stop 525
 if(any(ans.ne.23d0).or.any(unc.ne.31d0).or.any(rand.ne..13d0).or.any(icell.ne.2))stop 526
 if(any(nhits.ne.0).or.online_ordinary_calls.ne.64.or.online_probes.ne.1.or. &
      online_reused_nodes.ne.16)stop 527
 if(any(abs(xacc(1,1:4,1)-256d0).gt.1d-12))stop 528
 if(trim(action).eq.'empty_intervening')then
  if(online_slots(1)%loss_before.ne.16d0.or.online_slots(1)%loss_after.ne.16d0)stop 529
  print *,'PASS empty intervening epoch removes stale history exactly'
 else
  if(online_slots(1)%loss_before.ne.48d0.or.online_slots(1)%loss_after.ne.48d0)stop 530
  print *,'PASS mixed diagnostic-only loss with CURRENT support/counters/proxy and ordinary isolation'
 endif
endif
end program
double precision function ran2()
call random_number(ran2)
end function
'''


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestEpochReplayFortran(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='mg7-epoch-replay-math-')
        cls.addClassCleanup(cls.temporary.cleanup);cls.work = Path(cls.temporary.name)
        support = mint_support(cls.work, real_fks_params=True)
        exposed = cls.work / 'exposed_mint.f90';exposed.write_text((TEMPLATE / 'mint_module.f90').read_text().replace('private', 'public'))
        support = [str(exposed) if path == str(TEMPLATE / 'mint_module.f90') else path for path in support]
        program = cls.work / 'check.f90';program.write_text(PROGRAM);cls.executable = cls.work / 'check'
        result = subprocess.run(['gfortran', '-O0', '-fcheck=all', '-ffree-line-length-none', '-ffixed-line-length-132',
            '-I', str(cls.work), *support, str(program), '-o', str(cls.executable)], cwd=cls.work, capture_output=True, text=True)
        if result.returncode:raise AssertionError(result.stdout + result.stderr)

    def run_case(self, action, failure=None):
        """Run one fixture action; a failing action must stop at its own guard."""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([str(self.executable), action], cwd=directory, capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
        if failure is None:
            self.assertEqual(result.returncode, 0, output);self.assertIn('PASS', output)
        else:
            self.assertNotEqual(result.returncode, 0, output);self.assertNotIn('PASS', output)
            self.assertIn(failure, output)

    def test_scaled_current_first_canonical_padding_and_immutability(self):self.run_case('normal')
    def test_empty_previous_exact_maps_losses_and_traces(self):self.run_case('empty_previous')
    def test_empty_current_does_not_replay_old_observations(self):self.run_case('empty_current')
    def test_current_support_not_increased_and_old_points_never_update_proxy_or_integral(self):self.run_case('runtime')
    def test_positive_history_cannot_fit_empty_current_epoch(self):self.run_case('runtime_empty')
    def test_empty_intervening_epoch_clears_history(self):self.run_case('empty_intervening')
    def test_nonfinite_and_incompatible_shapes_rejected(self):
        self.run_case('nan', 'Nonfinite epoch replay bank');self.run_case('badshape', 'Invalid epoch replay bank shapes')
