"""Check specialized history weights and their ISR sampling measure.

Controlled matrix elements and MC kernels isolate the production FKS
prefactors and S/H evaluator from PDF and loop-library dependencies.
"""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestMCHistoryWeights(unittest.TestCase):
    def test_h_and_s_paths_preserve_native_weights(self):
        with tempfile.TemporaryDirectory(prefix='mg5_mc_history_weights_') as tmp:
            work = Path(tmp)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\n'
                                 'parameter(nexternal=5,nincoming=2)',
                'genps.inc': 'integer maxproc,max_branch,max_particles\n'
                             'parameter(maxproc=1,max_branch=8,max_particles=8)',
                'coupl.inc': 'double precision g,test_mass\ncommon/test_couplings/g\n'
                             'common/test_mass/test_mass',
                'orders.inc': 'integer nsplitorders,amp_split_size,qcd_pos,qed_pos,cpower_pos\n'
                              'parameter(nsplitorders=2,amp_split_size=3)\n'
                              'parameter(qcd_pos=1,qed_pos=2,cpower_pos=0)\n'
                              'double precision amp_split(3)\n'
                              'common/test_amplitudes/amp_split',
                'pmass.inc': 'pmass=0d0\npmass(3)=test_mass',
                'run.inc': 'integer ickkw,lpp(2)\n'
                           'double precision scale,q2fact(2),xbk(2)\n'
                           'double precision real_channels_x0\nparameter(real_channels_x0=0d0)\n'
                           'common/test_run/scale,q2fact,xbk,ickkw,lpp',
                'q_es.inc': 'double precision QES2\ncommon/test_qes/QES2',
                'fks_info.inc': 'integer fks_i_d(1)\nparameter(fks_i_d=[5])',
            }
            for name, source in includes.items():
                (work / name).write_text(''.join('      '+s+'\n' for s in source.splitlines()))
            for name in ('timing_variables.inc', 'fks_powers.inc'):
                shutil.copyfile(TEMPLATE / name, work / name)
            (work / 'contexts.f90').write_text(
                'module extra_weights\n'
                'integer :: QCD_power,orders_tag,amp_pos\n'
                'double precision :: wgtcpower\nend module\n'
                'module mc_native_context\n'
                'integer :: active_history=0,native_provider_ids(1)=1,native_context_ids(1)=1\n'
                'logical :: native_mapping=.false.\n'
                'end module\n'
                'module FKSParams\n'
                'integer :: VetoedContributionTypes(0:1)=0,SelectedContributionTypes(0:1)=0\n'
                'integer :: SelectedCouplingOrders(2,0:1)=0\n'
                'integer :: QCD_squared_selected=-1,QED_squared_selected=-1\n'
                'contains\ninteger function get_fks_isr_mapping(nlo_ps,shower) result(mapping)\n'
                'logical nlo_ps\ncharacter(*) shower\nmapping=2\nend function\nend module\n'
                'module mint_module\nlogical :: pass_cuts_check=.false.,fixed_order=.false.\nend module\n'
                'module mc_counterterms\n'
                'double precision :: gfactsf,gfactcl,gfactazi\nend module\n')
            routines = ('compute_native_NLOPS_weights', 'compute_real_emission',
                        'compute_soft_counter_term', 'compute_collinear_counter_term',
                        'compute_soft_collinear_counter_term', 'add_wgt',
                        'compute_prefactors_n1body')
            (work / 'weights.f').write_text('\n'.join(
                fortran_routine(TEMPLATE / 'fks_singular.f', name) for name in routines) +
                '\n      double precision function real_channel_soft_fraction(p)\n'
                '      double precision p(0:3,5)\n'
                '      real_channel_soft_fraction=1d0\n      end\n')
            (work / 'measure.f').write_text(
                '      module history_sampling_measure\n'
                '      use fks_phase_space_helpers, only: lambda\n'
                '      use fks_phase_space_data, only: p1_cnt,jac_cnt\n'
                "      include 'nexternal.inc'\n"
                '      double precision counter_y(-2:2)\n'
                '      contains\n' +
                ''.join(fortran_routine(TEMPLATE / 'genps_fks.f', name)
                        for name in ('compute_flux', 'fill_fks_point_data')) +
                '      end module\n')
            executable = work / 'check_history_weights'
            result = subprocess.run([
                shutil.which('gfortran'), str(TEMPLATE.parent / 'Source/timing.f'), '-O2', '-std=legacy', '-fcheck=all',
                '-ffixed-line-length-none', '-ffunction-sections', '-fdata-sections',
                '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections',
                '-I', str(work),
                str(TEMPLATE.parent / 'Source/BornSupport/mc_born_types.f90'),
                str(TEMPLATE / 'fks_phase_space_data.f'),
                str(work / 'contexts.f90'), str(TEMPLATE / 'genps_fks_helpers.f'),
                str(TEMPLATE / 'genps_fks_radiation.f'), str(work / 'measure.f'),
                str(TEMPLATE / 'weight_lines.f'),
                str(work / 'weights.f'), str(TEMPLATE / 'boostwdir2.f'),
                str(ROOT / 'HELAS/boostx.F'), str(ROOT / 'HELAS/rotxxx.F'),
                str(ROOT / 'tests/input_files/check_mc_history_weights.f90'),
                '-o', str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for matching in ('ordinary', 'fxfx', 'sampling', 'mapfxfx'):
                with self.subTest(matching=matching):
                    result = subprocess.run([str(executable), matching], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('PASS native H and S weights', result.stdout)
