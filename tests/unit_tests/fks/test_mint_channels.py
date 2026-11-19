"""Exercise MINT channel allocation and integration with controlled integrands."""

import pathlib
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.mint_fixture import mint_runtime_modules


ROOT = pathlib.Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class TestMintChannels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='mg5-mint-channels-')
        cls.path = pathlib.Path(cls.directory.name)
        source = (ROOT / 'Template/NLO/SubProcesses/mint_module.f90').read_text()
        # Expose state to the test driver without changing the production API.
        source = source.replace('private', 'public')
        (cls.path / 'mint_module.f90').write_text(source)
        (cls.path / 'support.f90').write_text(SUPPORT)
        (cls.path / 'check.f90').write_text(PROGRAM)
        result = subprocess.run(
            ['gfortran', '-O0', '-g', '-fcheck=all', '-ffree-line-length-none',
             '-o', 'check', 'support.f90', str(ROOT / 'Template/NLO/SubProcesses/virtual_surrogate.f90'),
             *mint_runtime_modules(cls.path),
             'mint_module.f90', 'check.f90'],
            cwd=cls.path, capture_output=True, text=True)
        if result.returncode:
            cls.directory.cleanup()
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_case(self, case, directory=None):
        result = subprocess.run([str(self.path / 'check'), case],
                                cwd=directory or self.path, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return [line.split()[1:] for line in result.stdout.splitlines()
                if line.startswith(' CHECK ')]

    def test_zero_virtual_envelope_bootstraps_from_finite_positive_iteration(self):
        self.run_case('virtual_envelope')

    def test_virtual_envelope_bootstrap_averages_an_isolated_outlier(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_case('envelope_seed', directory)
            self.run_case('envelope_recover', directory)

    def test_rms_allocation_and_unbiased_channel_compensation(self):
        probability, integral = self.run_case('rms')
        probability = list(map(float, probability))
        # The two relevant channels have equal integrals, but RMS weights 1:10.
        self.assertAlmostEqual(probability[0], .95 / 11 + .05 / 3)
        self.assertAlmostEqual(probability[1], .95 * 10 / 11 + .05 / 3)
        # A tiny channel with a 100% pilot error gets only its exploration share.
        self.assertAlmostEqual(probability[2], .05 / 3, delta=1.e-9)
        mean, error = map(float, integral)
        self.assertLess(abs(mean - (2 + 1.e-9)), 5 * error)

    def test_unvisited_channel_keeps_its_previous_moment(self):
        moment, calls = self.run_case('history')
        self.assertEqual(list(map(float, moment)), [4., 9.])
        self.assertEqual(list(map(int, calls)), [17, 5])

    def test_first_automatic_iteration_still_probes_each_channel(self):
        self.run_case('pilot')

    def test_mc_channel_selection_keeps_legacy_probabilities(self):
        counts, = self.run_case('mc')
        self.assertAlmostEqual(int(counts[0]) / sum(map(int, counts)), .875, delta=.006)

    def test_native_mint_draws_commit_in_order_including_short_tails(self):
        with tempfile.TemporaryDirectory() as refdir, tempfile.TemporaryDirectory() as work:
            expected = self.run_case('draw_scalar', refdir)
            self.assertEqual(len(expected), 1)
            self.assertEqual(self.run_case('draw_cohort', work), expected)

    def test_interleaved_folds_keep_their_cells_and_generation_envelope(self):
        self.run_case('fold_and_hit')

    def test_modern_and_legacy_native_job_headers(self):
        self.run_case('job_headers')

    def test_zero_moments_give_uniform_exploration(self):
        probability, = self.run_case('zero')
        self.assertEqual(list(map(float, probability)), [.5, .5])

    def test_automatic_pilot_updates_both_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self.run_case('automatic', directory)
            mean, error, second = map(float, result)
            self.assertLess(abs(mean - 3.), 5 * error)
            self.assertLess(abs(second - 1.5), .1)

    def test_restart_recovers_a_channel_with_zero_integral(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as directory:
                self.run_case('seed', directory)
                grid = pathlib.Path(directory) / 'mint_grids'
                if legacy:
                    grid.write_text(''.join(' '.join(line.split()[:4]) + '\n'
                                            if line.split()[0] == 'SPE' else line
                                            for line in grid.read_text().splitlines(True)))
                result, moments = self.run_case('recover', directory)
                mean, error, recovered = map(float, result)
                # Two channels each integrate to 1.5, with split multiplier .25.
                self.assertLess(abs(mean - .75), 5 * error)
                self.assertGreater(recovered, .3)
                # Stored moments exclude both channel probability and wgt_mult.
                self.assertTrue(all(2.1 < float(moment) < 2.6 for moment in moments))
                fields = [line.split() for line in grid.read_text().splitlines()
                          if line.split()[0] == 'SPE']
                self.assertTrue(all(len(row) == 6 and int(row[5]) > 0 for row in fields))


SUPPORT = '''
module FKSParams
logical :: use_poly_virtual=.false.
integer :: VirtTrickSurrogate=0
double precision :: virt_fraction=1d0,min_virt_fraction=.005d0
end module

double precision function ran2()
implicit none
call random_number(ran2)
end function
'''

# Plotting, virtual-fit and FKS-integer hooks are irrelevant to these integrands.
for _name in ('accum', 'HwU_accum_iter', 'HwU_add_points', 'initplot',
              'regrid_MC_integer', 'empty_MC_integer', 'reset_MC_grid',
              'init_polyfit', 'get_polyfit', 'add_point_polyfit', 'do_polyfit',
              'save_polyfit', 'restore_polyfit', 'timing_begin_point',
              'timing_end_point', 'deallocate_born_spread_lines'):
    SUPPORT += '\nsubroutine %s\nend subroutine\n' % _name


PROGRAM = '''
module integrand_state
logical :: seed_only=.false.
logical :: envelope_case=.false.,virtual_point=.false.
integer :: virtual_calls=0
logical :: saw_three=.false., saw_tail=.false.
double precision :: virtual_share=0d0
end module

program check_mint
use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
use mint_module
use integrand_state
implicit none
character(len=16) :: mode
integer :: i,j,nseed,counts(3)
integer :: unit,width,label,events
type(mint_draw) :: trials(3)
logical :: hits(3)
integer, allocatable :: seed(:)
double precision :: estimate,total,total2,weight,point,error,u
double precision, external :: integrand
call get_command_argument(1,mode)
call random_seed(size=nseed)
allocate(seed(nseed))
seed=12345
call random_seed(put=seed)
fixed_order=.true.
nlo_ps=.false.
nchans=2
iconfigs(1:3)=[1,2,3]
imode=0
nit=2
double_events=.false.
! Common integration setup; cases override only the settings they exercise.
ndim=4
ncalls0=20000
itmax=4
accuracy=0d0
n_ord_virt=0
only_virt=.false.
ifold=1
ifold_energy=ndim-2
ifold_yij=ndim-1
ifold_phi=ndim
min_virt_fraction_mint=.005d0
wgt_mult=.5d0
select case(mode)
case('job_headers')
 open(newunit=unit,status='scratch')
 write(unit,'(a)') '3  ! integration width'
 write(unit,'(a)') '3  ! generation width'
 write(unit,'(a)') '0 12 ! process label, event count'
 write(unit,'(a)') '0 24 ! legacy event job'
 rewind(unit)
 call read_mc_job_header(unit,1,width,label,events)
 if (width/=3) error stop 'integration width'
 call read_mc_job_header(unit,2,width,label,events)
 if (width/=3.or.label/=0.or.events/=12) error stop 'explicit generation width'
 call read_mc_job_header(unit,2,width,label,events)
 if (width/=1.or.label/=0.or.events/=24) error stop 'legacy generation pair'
 call read_mc_job_header(unit,0,width,label,events)
 if (width/=1) error stop 'legacy integration'
 close(unit)
case('virtual_envelope')
 imode=1
 ichan=1
 ncell=1
 ymax=1d0
 ymax_virt=0d0
 virtual_fraction(1)=0.25d0
 f=0d0;f(1)=1d0;f(5)=4d0
 call add_point_to_bounding_envelope
 if(ymax_virt(1).ne.0d0)error stop 'single point bootstrapped virtual envelope'
 ! The mean of f(5) already excludes the virtual_fraction enhancement.
 vtot=0d0;vtot(5,1)=4d0
 call bootstrap_virtual_envelopes
 if(ymax_virt(1).ne.4d0)error stop 'zero virtual envelope did not bootstrap'
 f(5)=48d0
 call add_point_to_bounding_envelope
 if(ymax_virt(1).ne.8d0)error stop 'positive virtual envelope growth changed'
 f(5)=0.5d0
 call add_point_to_bounding_envelope
 if(ymax_virt(1).ne.8d0)error stop 'positive virtual envelope decreased'
 ymax_virt=0d0;vtot(5,1)=0d0
 call bootstrap_virtual_envelopes
 if(ymax_virt(1).ne.0d0)error stop 'empty virtual envelope changed'
 vtot(5,1)=-1d0
 call bootstrap_virtual_envelopes
 if(ymax_virt(1).ne.0d0)error stop 'negative virtual mean bootstrapped'
 vtot(5,1)=ieee_value(0d0,ieee_quiet_nan)
 call bootstrap_virtual_envelopes
 if(ymax_virt(1).ne.0d0)error stop 'nonfinite virtual mean bootstrapped'
 vtot(5,1)=ieee_value(0d0,ieee_positive_inf)
 call bootstrap_virtual_envelopes
 if(ymax_virt(1).ne.0d0)error stop 'infinite virtual mean bootstrapped'
 vtot(5,1)=4d0
 imode=2
 call bootstrap_virtual_envelopes
 if(ymax_virt(1).ne.0d0)error stop 'virtual bootstrap ran during generation'
case('envelope_seed','envelope_recover')
 nchans=1
 envelope_case=.true.
 itmax=2
 wgt_mult=1d0
 ! One envelope iteration, so that its bootstrap seed is not grown further.
 if(mode.eq.'envelope_recover')then
  imode=1
  itmax=1
 endif
 call mint(integrand)
 if(mode.eq.'envelope_seed')then
  if(ans(5,1).ne.0d0)error stop 'virtual bootstrap fixture did not start at zero'
 else
  ! The seed is the iteration mean, as ans(5) is in setup_imode_1.
  if(ymax_virt(1).ne.ans(5,1))error stop 'virtual envelope bootstrap is not the mean'
  ! The outlier alone has an un-enhanced virtual of about 1e4.
  if(ymax_virt(1).le.0d0.or.ymax_virt(1).gt.1d2) &
    error stop 'isolated outlier dominated virtual envelope bootstrap'
 endif
case('rms')
 nchans=3
 ! The tiny channel has mean 1e-9 and RMS 1e-8: about 100% error
 ! after a pilot with 100 samples and one nonzero weight of 1e-7.
 channel_moment(1:3)=[1d0,100d0,1d-16]
 call set_channel_probabilities
 write(*,*) 'CHECK ',channel_probability(1:3)
 total=0d0
 total2=0d0
 do i=1,200000
  call get_channel
  point=1d0
  if(ichan.eq.2) then
   call random_number(u)
   point=0d0
   if(u.lt..01d0) point=100d0
  elseif(ichan.eq.3) then
   call random_number(u)
   point=0d0
   if(u.lt..01d0) point=1d-7
  endif
  weight=point/vol_chan
  total=total+weight
  total2=total2+weight**2
 enddo
 estimate=total/200000d0
 error=sqrt((total2/200000d0-estimate**2)/200000d0)
 write(*,*) 'CHECK ',estimate,error
case('history')
 channel_moment(1:2)=[4d0,100d0]
 channel_moment_calls(1:2)=[17,17]
 channel_calls(1:2)=[0,5]
 channel_squares(1:2)=[0d0,45d0]
 call update_channel_sampling
 write(*,*) 'CHECK ',channel_moment(1:2)
 write(*,*) 'CHECK ',channel_moment_calls(1:2)
case('pilot')
 nit=1
 double_events=.true.
 initial_channel_scan_pending=.true.
 channel_moment(1:2)=[0d0,100d0]
 call set_channel_probabilities
 do j=1,2
  ans_chan=0d0
  ans_chan(0)=1d0
  ans_chan(j)=1d0
  do i=1,100
   call get_channel
   if(ichan.ne.j.or.vol_chan.ne.1d0) stop 1
  enddo
 enddo
case('mc')
 fixed_order=.false.
 channel_moment(1:2)=[1d0,100d0]
 call set_channel_probabilities
 ans_chan(0:2)=[8d0,7d0,1d0]
 counts=0
 do i=1,100000
  call get_channel
  counts(ichan)=counts(ichan)+1
  if(abs(vol_chan-ans_chan(ichan)/8d0).gt.1d-15) stop 2
 enddo
 write(*,*) 'CHECK ',counts(1:2)
case('fold_and_hit')
 fixed_order=.false.
 nlo_ps=.true.
 nchans=1
 imode=2
 ifold(ndim)=2
 do i=0,nintervals
   xgrid(i,:ndim,1)=dble(i)/nintervals
 enddo
 ave_virt=0d0
 ymax=1d0
 ymax_virt(1)=4d0
 call initialise_mint_gen
 do i=1,3
   call generate_mc_trials(trials,fold_and_hit,i,hits)
   if (any(hits.neqv.[.true.,.false.,.true.])) error stop 'wrong trial accepted'
 enddo
 if (any(gen_counters(3:10)/=[9,3,3,6,2,2,2,3])) error stop 'wrong generation accounting'
case('draw_scalar','draw_cohort')
 fixed_order=.false.
 nlo_ps=.true.
 nchans=1
 ncalls0=131
 itmax=3
 if (mode=='draw_cohort') then
    call mint(integrand,run_cohort,3)
    if (.not.saw_three.or..not.saw_tail) error stop 'expected full cohorts and tails'
 else
    call mint(integrand)
 endif
 write(*,*) 'CHECK ',ans(2,0),unc(2,0),sum(xgrid(:,1,1))
case('zero')
 channel_moment=0d0
 call set_channel_probabilities
 write(*,*) 'CHECK ',channel_probability(1:2)
case('seed','recover','automatic')
 seed_only=mode.eq.'seed'
 if(mode.eq.'recover') then
  imode=-1
  wgt_mult=.25d0
 endif
 if(mode.eq.'automatic') then
  wgt_mult=1d0
  ncalls0=-1
  accuracy=.01d0
  itmax=6
 endif
 call mint(integrand)
 write(*,*) 'CHECK ',ans(2,0),unc(2,0),ans(2,2)
 write(*,*) 'CHECK ',channel_moment(1:2)
end select
contains
subroutine fold_and_hit(draws)
 type(mint_draw),intent(inout) :: draws(:)
 integer lane,step
 double precision original(size(draws)),envelope
 ! Virtual draws use the flat-cell virtual bound; the other streams
 ! use a product of per-dimension bounds.
 envelope=merge(4d0,1d0,i==1)
 do step=1,3
   do lane=size(draws),1,-1
      call activate_mint_draw(draws(lane))
      if (upper_bound/=envelope) error stop 'envelope changed across draws'
      if (draws(lane)%ifl/=step-1) error stop 'wrong native fold stage'
      if (step==1) then
         original(lane)=draws(lane)%coordinates(ndim)
      else
         if (abs(draws(lane)%coordinates(ndim)-original(lane)-0.5d0)>1d-15) error stop 'fold was redrawn'
         if (icell(ndim)/=ncell(ndim)+nintervals/2) error stop 'wrong folded cell'
      endif
      if (step==3) draws(lane)%integrals(1)=merge(0d0,2*envelope,lane==2)
      call finish_mint_fold(draws(lane))
      call park_mint_draw(draws(lane))
   enddo
 enddo
end subroutine

subroutine run_cohort(draws)
 use mc_decision_ledger, only: current_draw
 type(mint_draw),intent(inout) :: draws(:)
 integer lane
 saw_three=saw_three.or.size(draws)==3
 saw_tail=saw_tail.or.size(draws)<3
 do lane=1,size(draws)
   call activate_mint_draw(draws(lane))
   if (current_draw/=draws(lane)%id) error stop 'wrong draw'
   virt_wgt_mint(0)=dble(current_draw)
   allocate(mint_schannel_order(-3:0))
   mint_schannel_order=int(current_draw)*10+[0,1,2,3]
   pass_cuts_check=.true.
   call park_mint_draw(draws(lane))
 enddo
 do lane=size(draws),1,-1
   call activate_mint_draw(draws(lane))
   if (virt_wgt_mint(0)/=dble(draws(lane)%id)) error stop 'leaked virtual weights'
   if (lbound(mint_schannel_order,1)/=-3.or.mint_schannel_order(-2)/=int(current_draw)*10+1) &
        error stop 'leaked stochastic resonance ordering'
   if (any(ncell(:ndim)/=draws(lane)%base_cell(:ndim))) error stop 'leaked cells'
   call compute_integrand(integrand,draws(lane)%coordinates,draws(lane)%volume)
   draws(lane)%integrals=f
   draws(lane)%complete=.true.
   call park_mint_draw(draws(lane))
 enddo
end subroutine
end program

double precision function integrand(x,vol,ifl,f)
use mint_module, only: ndimmax,nintegrals,ichan,pass_cuts_check,imode,virtual_fraction
use integrand_state
implicit none
integer :: ifl
double precision :: x(ndimmax),vol,f(nintegrals),point
point=1d0+x(1)
if(seed_only.and.ichan.eq.2) point=0d0
f=0d0
f(1)=point*vol
f(2)=f(1)
if(envelope_case.and.imode.eq.1)then
 ! As in the driver, the virtual is evaluated for a virtual_fraction of
 ! the points only, enhanced by its inverse. The first one is an outlier.
 if(ifl.eq.0)then
  virtual_share=virtual_share+virtual_fraction(ichan)
  virtual_point=virtual_share.ge.1d0
  if(virtual_point)virtual_share=virtual_share-1d0
 endif
 if(virtual_point)then
  f(5)=vol/virtual_fraction(ichan)
  if(virtual_calls.eq.0)f(5)=1d4*f(5)
  if(ifl.eq.2)virtual_calls=virtual_calls+1
 endif
endif
pass_cuts_check=.true.
integrand=f(2)
end function
'''
