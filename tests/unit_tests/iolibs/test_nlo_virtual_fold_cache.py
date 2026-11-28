"""Exercise the production virtual cache with interleaved lanes and folds."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from madgraph import MG5DIR


class TestVirtualFoldCache(unittest.TestCase):
    def test_grid_training_restores_lane_approximation(self):
        compiler = shutil.which('gfortran')
        if not compiler:
            self.skipTest('gfortran required')
        source = (Path(MG5DIR)/'Template/NLO/SubProcesses/mint_module.f90').read_text()
        start = source.index('             call restore_cells(ivec)')
        restore = source[start:source.index('enddo', start)]
        start = source.index('  subroutine add_point_to_grids(x)')
        training = source[start:source.index('  end subroutine add_point_to_grids', start)]
        training += '  end subroutine add_point_to_grids\n'
        probe = '''module probe_state
implicit none
integer, parameter :: ndimmax=4,ndim=4,n_ord_virt=1,ichan=1
integer ivec,icell(4),mode
logical use_poly_virtual,pass_cuts_check,pass_cuts_check_vec(2)
double precision x(4),x_mint_vec(4,2),f(8),f_vec(8,2),xacc(1,4,1)
double precision virt_wgt_mint(0:1),born_wgt_mint(0:1)
double precision virt_wgt_vec(0:1,2),born_wgt_vec(0:1,2)
double precision average_virtual(0:1,1),polyfit(0:1)
double precision average_virtual_vec(0:1,2),polyfit_vec(0:1,2)
double precision virtual_fraction(1),wgt_mult,observed(0:1,2)
! virt_surrogate state: no fitted surrogate for this point.
logical :: vs_vcomp=.false.,vs_ratio_set=.false.
double precision :: vs_ratio=0d0
contains
subroutine accumulate_the_point(x)
double precision x(4)
call add_point_to_grids(x)
end subroutine
subroutine restore_cells(lane)
integer lane
icell=1
end subroutine
subroutine fill_ave_virt(x,k,v,b)
integer k
double precision x(4),v,b
observed(k,ivec)=v
end subroutine
subroutine add_point_polyfit(channel,k,x,v,b)
integer channel,k
double precision x(1),v,b
observed(k,ivec)=v*b
end subroutine
''' + training + '''end module
program main
use probe_state
implicit none
virtual_fraction=.5d0
wgt_mult=1d0
x_mint_vec=0d0
virt_wgt_vec=0d0
born_wgt_vec=0d0
pass_cuts_check_vec=.true.
average_virtual_vec(:,1)=2d0
average_virtual_vec(:,2)=5d0
polyfit_vec=average_virtual_vec
f_vec=0d0
f_vec(6,:)=3d0
f_vec(8,:)=7d0
f_vec(3,:)=(12d0-average_virtual_vec(0,:)*f_vec(6,:))/.5d0
f_vec(7,:)=(28d0-average_virtual_vec(1,:)*f_vec(8,:))/.5d0
do mode=0,1
use_poly_virtual=mode.eq.1
average_virtual(:,1)=average_virtual_vec(:,2)
polyfit=polyfit_vec(:,2)
observed=0d0
xacc=0d0
do ivec=1,2
''' + restore + '''enddo
if (any(observed(0,:) /= 12d0)) stop 1
if (any(observed(1,:) /= 28d0)) stop 2
enddo
end program
'''
        with tempfile.TemporaryDirectory(prefix='mg7_lane_approx_') as tmp:
            tmp = Path(tmp)
            (tmp/'probe.f90').write_text(probe)
            compiled = subprocess.run([compiler, '-O0', '-fcheck=all', 'probe.f90',
                                       '-o', 'probe'], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([str(tmp/'probe')], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_cache_lifecycle_survives_fold_reset(self):
        # The vector storage is zeroed when allocated, kept by reset_storage
        # (a fold reset) and freed by deallocate_storage. The native MC driver
        # no longer clears it between folds: each lane's virtual is restored
        # from its own context instead (see the bornsoftvirtual test below).
        compiler = shutil.which('gfortran')
        if not compiler:
            self.skipTest('gfortran required')
        root = Path(MG5DIR)/'Template/NLO/SubProcesses'
        clear = ('  bsv_virt_cache=0d0\n  virt_wgt_vec=0d0\n'
                 '  born_wgt_vec=0d0\n  virtual_over_born_vec=0d0')
        program = '''program lifecycle
use driver_vec
implicit none
integer n
do n=2,3
  call allocate_storage(n,4,2,6,2)
  if (any(bsv_virt_cache /= 0d0)) stop 1
  bsv_virt_cache(:,1)=11d0
  bsv_virt_cache(:,2)=29d0
  virt_wgt_vec=11d0
  born_wgt_vec=29d0
  virtual_over_born_vec=42d0
  call reset_storage()
  if (any(bsv_virt_cache(:,1) /= 11d0)) stop 2
  if (any(bsv_virt_cache(:,2) /= 29d0)) stop 3
  if (any(virt_wgt_vec /= 11d0)) stop 6
  if (any(born_wgt_vec /= 29d0)) stop 7
  if (any(virtual_over_born_vec /= 42d0)) stop 8
''' + clear + '''
  if (any(bsv_virt_cache /= 0d0)) stop 4
  if (any(virt_wgt_vec /= 0d0)) stop 9
  if (any(born_wgt_vec /= 0d0)) stop 10
  if (any(virtual_over_born_vec /= 0d0)) stop 11
  call deallocate_storage()
  if (allocated(bsv_virt_cache)) stop 5
enddo
end program
'''
        includes = {'nexternal.inc': 'integer, parameter :: nexternal=5',
                    'nFKSconfigs.inc': 'integer, parameter :: fks_configs=3',
                    'genps.inc': 'integer, parameter :: ngraphs=2,ncolor=2',
                    'orders.inc': 'integer, parameter :: amp_split_size=2,nsplitorders=2',
                    'born_nhel.inc': 'integer, parameter :: max_bhel=4'}
        with tempfile.TemporaryDirectory(prefix='mg7_fold_lifecycle_') as tmp:
            tmp = Path(tmp)
            for name, text in includes.items():
                (tmp/name).write_text(text+'\n')
            (tmp/'driver.f90').write_text((root/'driver.f90').read_text())
            (tmp/'probe.f90').write_text(program)
            compiled = subprocess.run([compiler, '-O0', '-fcheck=all', 'driver.f90',
                                       'probe.f90', '-o', 'probe'], cwd=tmp,
                                      capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            subprocess.check_call([str(tmp/'probe')], cwd=tmp)

    def test_folded_lanes_keep_their_virtual_components(self):
        """The first evaluation of a point computes and saves the virtual
        residual; later folds restore it. The saved values are per lane (the
        native scheduler swaps them with each lane's context), so the probe
        passes them in as lane storage."""
        compiler = shutil.which('gfortran')
        if not compiler:
            self.skipTest('gfortran required')
        source = (Path(MG5DIR)/'Template/NLO/SubProcesses/fks_singular.f').read_text()
        bsv = source[source.index('      subroutine bornsoftvirtual('):]
        start = bsv.index('      if (first_virtual_evaluation) then\n         vs_vcomp=.false.')
        block = bsv[start:bsv.index("      if (abrv(1:4).ne.'virt') then", start)]
        probe = '''      program probe
      implicit none
      integer first,lane,phase,ncalls
      common /probe_calls/ncalls
      double precision save_v(2),save_amp(2,2),input,v,c(2)
      double precision fraction,average
      character*4 abrv
      save_v=0d0
      save_amp=0d0
      ncalls=0
 10   read(*,*,end=20) first,lane,input,abrv,fraction,phase,average
      if (first.eq.-1) then
         save_v=0d0
         save_amp=0d0
         goto 10
      endif
      call evaluate(first.eq.1,input,abrv,fraction,phase,average,
     $              save_v(lane),save_amp(:,lane),v,c)
      write(*,*) v,c,ncalls
      goto 10
 20   end

      subroutine evaluate(first_virtual_evaluation,input,abrv,fraction,
     $     born_spread_phase,average,virt_wgt_save,amp_split_virt_save,
     $     result,component)
      implicit none
      integer amp_split_size,ichan,iamp,born_spread_phase
      parameter(amp_split_size=2)
      integer timing_virtual
      parameter(timing_virtual=4)
      logical first_virtual_evaluation,timing_sampled,use_surrogate
      logical vs_vcomp,vs_ratio_set,virtual_fold_ready
      character*4 abrv
      double precision input,fraction,average,result,component(2)
      double precision virt_wgt_save,amp_split_virt_save(amp_split_size)
      double precision p_born(0:3,1),born_wgt,virt_wgt,virtual_over_born
      double precision amp_split_virt(amp_split_size)
      double precision amp_split_born_for_virt(amp_split_size)
      double precision ave_ratio(0:amp_split_size),virtual_fraction(1)
      double precision vs_born,vs_fvirt,vs_virt,vs_residual_reference
      integer vs_residual_order
      double precision amp_split_finite_ML(amp_split_size)
      common /to_amp_split_finite/amp_split_finite_ML
      double precision ran2
      external ran2
      ichan=1
      timing_sampled=.false.
      use_surrogate=.false.
      vs_ratio_set=.false.
      born_wgt=1d0
      p_born=0d0
      p_born(0,1)=input
      virt_wgt=0d0
      amp_split_virt=0d0
      amp_split_born_for_virt=1d0
      ave_ratio=average
      virtual_fraction=fraction
''' + block + '''      result=virt_wgt
      component=amp_split_virt
      end
      double precision function ran2()
      ran2=0.5d0
      end
      subroutine timing_enter(category)
      integer category
      end
      subroutine timing_leave(category)
      integer category
      end
      subroutine binothlha_frame(p,b,v)
      implicit none
      integer ncalls
      common /probe_calls/ncalls
      double precision p(0:3,1),b,v
      double precision amp_split_finite_ML(2)
      common /to_amp_split_finite/amp_split_finite_ML
      ncalls=ncalls+1
      v=2*p(0,1)
      amp_split_finite_ML(1)=-p(0,1)/2
      amp_split_finite_ML(2)=3*p(0,1)/2
      end
'''
        steps, expected = [], []
        calls = 0
        for abrv, fraction, phase, average in (
                ('all', 1., 0, 0.), ('all', .5, 0, 2.), ('novi', 1., 0, 0.),
                ('all', .25, 0, 0.), ('virt', .25, 0, 0.),
                ('all', 1., 1, 0.), ('all', 1., 2, 0.), ('all', 1., 3, 1.5)):
            steps.append('-1 1 0 all 1 0 0')
            cache = {}
            # Repeated first evaluations, unequal lanes and interleaved folds.
            for first, lane in ((1, 1), (0, 1), (1, 1), (1, 2),
                                (0, 1), (0, 2), (0, 1), (0, 2)):
                value = 12. if lane == 1 else 28.
                steps.append(f'{first} {lane} {value} {abrv} {fraction} {phase} {average}')
                if first:
                    computed = phase not in (1, 2) and (
                        (fraction >= .5 and abrv[:3] != 'nov') or abrv == 'virt')
                    if computed:
                        calls += 1
                        parts = [-value/2-average, 3*value/2-average]
                        divisor = fraction if abrv != 'virt' else 1.
                        cache[lane] = [sum(parts)/divisor] + [p/divisor for p in parts]
                    else:
                        cache[lane] = [0., 0., 0.]
                expected.append(cache[lane] + [float(calls)])
            # A lane filled before the reset and only restored after it cannot leak.
            steps.extend(['-1 1 0 all 1 0 0', '0 2 0 all 1 0 0'])
            expected.append([0., 0., 0., float(calls)])
        with tempfile.TemporaryDirectory(prefix='mg7_fold_cache_') as tmp:
            path = Path(tmp)/'probe.f'
            path.write_text(probe)
            binary = Path(tmp)/'probe'
            compiled = subprocess.run([compiler, '-O0', '-fcheck=all',
                                       '-ffixed-line-length-132', str(path), '-o', str(binary)],
                                      capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            output = subprocess.check_output([str(binary)], input='\n'.join(steps)+'\n', text=True)
            actual = [[float(x) for x in line.split()] for line in output.splitlines()]
            self.assertEqual(actual, expected)
