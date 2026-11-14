"""Reuse physical Born samples while regenerating folded radiation endpoints."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import (
    ROOT, TEMPLATE, FKS_MAIN_ROUTINES, fks_test_module, fortran_routine)


STUBS = '''module born_sampler_fixture
integer :: samples=0,initializations=0
end module
subroutine generate_momenta_born(x,shat,sqrtshat,totmass,m,s,qmass,qwidth,mb,xpswgt,xjac)
use born_sampler_fixture
use fks_phase_space_data,only: p_born_l
implicit none
real(8) x(99),shat,sqrtshat,totmass,m(-8:8),s(-8:8),qmass(-5:0),qwidth(-5:0),mb(4),xpswgt,xjac
real(8) energy,momentum,angle
samples=samples+1
energy=(shat+mb(3)**2-mb(4)**2)/(2d0*sqrtshat)
momentum=sqrt(energy**2-mb(3)**2)
angle=6.283185307179586d0*x(1)
p_born_l(:,1)=[sqrtshat/2d0,0d0,0d0,sqrtshat/2d0]
p_born_l(:,2)=[sqrtshat/2d0,0d0,0d0,-sqrtshat/2d0]
p_born_l(:,3)=[energy,cos(angle)*momentum,sin(angle)*momentum,0d0]
p_born_l(:,4)=p_born_l(:,1)+p_born_l(:,2)-p_born_l(:,3)
xpswgt=1d0
end subroutine
subroutine generate_tau_y_wrapper(qmass,qwidth,totmass,stot,coords,tau,ycm,xjac)
implicit none
real(8) qmass(-5:0),qwidth(-5:0),totmass,stot,coords(2),tau,ycm,xjac
tau=1d0
ycm=0d0
xjac=1d0
end subroutine
subroutine generate_noevpr_kinematics
error stop 'unexpected coupled-lepton sampling'
end subroutine
'''


INITIALIZER = '''      subroutine initialize_fks_phase_space(iconfig)
      use born_sampler_fixture
      implicit none
      integer iconfig,this_config
      common/to_mconfigs/this_config
      this_config=iconfig
      initializations=initializations+1
      end subroutine initialize_fks_phase_space
'''


PROGRAM = '''program check
use fks_phase_space
use fks_phase_space_data,only: p_born_l,initial_recoil_leg, &
     tau_Born_lower_bound,tau_lower_bound_resonance,tau_lower_bound
use born_sampler_fixture
use mc_native_context
implicit none
integer,parameter :: ndim=6
integer :: sector,ifks,jfks,config,before
common/c_nFKSprocess/sector
common/fks_indices/ifks,jfks
real(8) :: ebeam(2),xbk(2),pmass(5),qmass(-5:0),qwidth(-5:0),x(99),w,wfresh
integer :: lpp(2)
common/test_run/ebeam,xbk,lpp
common/to_mass/pmass
common/c_qmass_qwidth/qmass,qwidth
logical softtest,colltest
common/sctests/softtest,colltest
real(8) saved_born(0:3,4)
type(fks_phase_space_point) point,fresh
softtest=.false.
colltest=.false.
lpp=1
ebeam=500d0
xbk=1d0
pmass=[0d0,0d0,173d0,300d0,0d0]
qmass=0d0
qwidth=0d0
sector=1
ifks=5
jfks=3
config=1
initial_recoil_leg=0
tau_Born_lower_bound=0d0
tau_lower_bound_resonance=0d0
tau_lower_bound=0d0
native_mapping=.false.
native_epoch=0
x=0d0
x(1:6)=[0.2d0,0.3d0,0.4d0,0.999d0,0.999d0,0.137d0]
w=2d0
call generate_born_contribution(ndim,config,w,x,point)
if(samples.ne.1.or..not.point%born%valid)error stop 'initial Born not sampled'
if(.not.point%no_counterevents.or.point%counterevent(0)%valid)error stop 'first fold has endpoint'
saved_born=point%born%p
! The cached Born survives even though the first radiation fold has no
! soft endpoint. A later fold must regenerate and gain that endpoint.
x(4:5)=modulo(x(4:5)+0.5d0,1d0)
w=3d0
call generate_born_contribution(ndim,config,w,x,point)
if(samples.ne.1.or.any(point%born%p.ne.saved_born))error stop 'fold resampled Born'
if(point%no_counterevents.or..not.point%counterevent(0)%valid)error stop 'fold reused stale endpoints'
w=5d0
call generate_real_phase_space(ndim,config,w,x,point)
if(samples.ne.1)error stop 'real wrapper resampled reusable Born'
call compare_fresh()
! Each guard below changes a provenance component while retaining the
! old point. The real preparation routine must sample once, then permit
! radiation-only reuse. The sampler uses masses, energy and x(1), so
! those changes also alter physical Born momenta and endpoint measures.
x(1)=0.27d0
call resamples_once()
pmass(3)=180d0
call resamples_once()
ebeam=600d0
call resamples_once()
lpp=0
call resamples_once()
sector=2
call resamples_once()
config=2
call resamples_once()
native_epoch=1
call resamples_once()
tau_Born_lower_bound=0.01d0
call resamples_once()
tau_lower_bound_resonance=0.02d0
call resamples_once()
tau_lower_bound=0.03d0
call resamples_once()
! A new complete point invalidates the cached input even if random
! coordinates happen to coincide with its predecessor.
point%born%valid=.false.
call resamples_once()
point%born%sampled=.false.
call resamples_once()
if(initializations.le.samples)error stop 'context not initialized for cached calls'
write(*,*)'PASS folded Born cache; Born sampler calls:',samples
contains
subroutine compare_fresh()
before=samples
fresh=fks_phase_space_point()
wfresh=5d0
call generate_real_phase_space(ndim,config,wfresh,x,fresh)
if(samples.ne.before+1)error stop 'forced baseline was cached'
if(any(point%born%p.ne.fresh%born%p).or.any(point%p.ne.fresh%p).or. &
   any(point%p_lab.ne.fresh%p_lab).or.any(point%p_cms.ne.fresh%p_cms))error stop 'cached momenta changed'
if(w.ne.wfresh.or.point%weight.ne.fresh%weight.or. &
   point%counterevent(0)%jacobian.ne.fresh%counterevent(0)%jacobian.or. &
   any(point%counterevent(0)%p.ne.fresh%counterevent(0)%p))error stop 'cached measure changed'
end subroutine
subroutine resamples_once()
integer init_before
init_before=initializations
before=samples
w=5d0
call generate_real_phase_space(ndim,config,w,x,point)
if(samples.ne.before+1.or..not.point%born%valid)error stop 'changed context reused stale Born'
if(initializations.ne.init_before+1)error stop 'miss skipped context initialization'
x(6)=modulo(x(6)+0.17d0,1d0)
w=5d0
call generate_real_phase_space(ndim,config,w,x,point)
if(samples.ne.before+1)error stop 'radiation changed Born cache key'
if(initializations.ne.init_before+2)error stop 'cache hit skipped context initialization'
call compare_fresh()
end subroutine
end program
'''


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestLearningProbeBornCache(unittest.TestCase):
    def test_born_wrappers_reuse_samples_and_regenerate_physical_endpoints(self):
        with tempfile.TemporaryDirectory(prefix='mg5_folding_born_') as tmp:
            work = Path(tmp)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\nparameter(nexternal=5,nincoming=2)',
                'genps.inc': 'integer max_branch,max_particles\nparameter(max_branch=8,max_particles=8)',
                'orders.inc': 'integer nsplitorders\nparameter(nsplitorders=1)',
                'run.inc': 'real(8) ebeam(2),xbk(2)\ninteger lpp(2)\ncommon/test_run/ebeam,xbk,lpp',
                'coupl.inc': '',
                'pmass.inc': 'common/to_mass/pmass',
            }
            for name, body in includes.items():
                (work / name).write_text(''.join('      ' + line + '\n' for line in body.splitlines()))
            for name in ('timing_variables.inc', 'fks_powers.inc'):
                shutil.copyfile(TEMPLATE / name, work / name)
            (work / 'native_context.f90').write_text(
                'module mc_native_context\nlogical :: native_mapping=.false.\n'
                'integer :: native_epoch=0\nend module\n')
            (work / 'sampler.f90').write_text(STUBS)
            names = FKS_MAIN_ROUTINES + (
                'generate_born_contribution', 'generate_real_phase_space',
                'generate_prepared_fks_point', 'prepare_fks_beam_point',
                'sample_prepared_fks_born', 'fks_born_matches')
            module = fks_test_module(names)
            module = module.replace('      end module fks_phase_space', INITIALIZER + '      end module fks_phase_space')
            (work / 'phase_space.f').write_text(module)
            routines = []
            for name in ('rotate_invar', 'trp_rotate_invar', 'phspncheck_nocms', 'xlen4',
                         'xmom_compare', 'xmcompare', 'xprintout'):
                routines.append(fortran_routine(TEMPLATE / 'fks_singular.f', name))
            for name in ('dot', 'rho', 'threedot'):
                routines.append(fortran_routine(TEMPLATE.parent / 'Source/kin_functions.f', name))
            (work / 'maps.f').write_text('\n'.join(routines))
            (work / 'check.f90').write_text(PROGRAM)
            command = [shutil.which('gfortran'), '-O2', '-std=legacy', '-ffixed-line-length-none',
                       '-ffunction-sections', '-fdata-sections', '-fcheck=all',
                       '-ffpe-trap=invalid,zero,overflow',
                       '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections',
                       '-I', str(work), str(TEMPLATE.parent / 'Source/timing.f'),
                       str(TEMPLATE / 'fks_phase_space_data.f'),
                       str(TEMPLATE / 'genps_fks_helpers.f'), str(work / 'native_context.f90'),
                       str(TEMPLATE / 'FKSParams.f90'), str(TEMPLATE / 'mcatnlo_delta_scales.f90'),
                       str(TEMPLATE / 'genps_fks_radiation.f'), str(work / 'sampler.f90'),
                       str(work / 'phase_space.f'), str(work / 'maps.f'),
                       str(TEMPLATE / 'boostwdir2.f'), str(TEMPLATE / 'resonance_recoil.f'),
                       str(TEMPLATE / 'initial_recoil.f'), str(ROOT / 'HELAS/boostx.F'),
                       str(ROOT / 'HELAS/rotxxx.F'), str(work / 'check.f90'), '-o', str(work / 'check')]
            result = subprocess.run(command, cwd=work, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(work / 'check')], cwd=work, text=True,
                                    capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS folded Born cache', result.stdout)
