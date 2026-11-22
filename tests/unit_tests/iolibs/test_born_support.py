"""Registry invariants and compiled equivalence with the standalone Born code."""

import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from madgraph import MadGraph5Error
from madgraph.iolibs import born_support as support
from madgraph.interface.common_run_interface import CommonRunCmd
from tests.unit_tests.fks.test_momentum_maps import mc_counterterm_test_module
from tests.unit_tests.iolibs import model_state_checks


def write_real_cache_fixture(root, legs, slots):
    """Use the generated cache representation and lookup in the isolated scalar router."""
    templates = Path(support.__file__).parent/'template_files'
    native = Path(support.__file__).resolve().parents[2]/'Template/NLO/SubProcesses'
    types = (templates/'mc_native_provider_types.inc').read_text()
    methods = (templates/'mc_native_provider_methods.inc').read_text()
    lookup = re.search(r'(?ms)^integer function mc_real_cache_lookup\b.*?^end function', methods).group()
    lookup = re.search(r'(?ms)^subroutine mc_ensure_real_cache\b.*?^end subroutine', methods).group()+'\n'+lookup
    (root/'mc_native_context.f90').write_text(f"""module mc_native_context
use mc_born_types
implicit none
integer,parameter :: provider_legs={legs},provider_orders={slots}
logical,save :: shared_real_active=.true.
integer,save :: shared_real_epoch=1,active_history=0
real(8),save :: shared_real_point(0:3,provider_legs)=1d0
integer,save :: history_permutations(provider_legs,1)=0
{types}
contains
{lookup}
end module
""")
    return [str(native/'mc_batch_trace.f90'),str(native/'mc_decision_ledger.f90')]


def physical(ids, tags=None):
    tags = tags or [False]*len(ids)
    return ('restrictions', tuple((pdg, i >= 2, (), tag, None)
                                 for i,(pdg,tag) in enumerate(zip(ids,tags))))


def context(number, processes, i, j, allowed, orders=('QCD','QED')):
    return dict(directory='P%d' % number, provider=number, context=number,
                nexternal=len(processes[0][1]), nincoming=2, orders=orders,
                amp_orders=[(2,2)], sectors=[dict(sector=1,
                    processes=processes, allowed=allowed,
                    fks=dict(i=i, j=j, splitting_type=['QCD'], extra_cnt_index=-1))])


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestStandaloneSudakov(unittest.TestCase):
    def test_equivalent_subprocesses_keep_standalone_born_sources(self):
        from madgraph.interface.master_interface import MasterCmd
        with tempfile.TemporaryDirectory(prefix='mg5_sudakov_export_') as tmp:
            output = Path(tmp)/'output'
            interface = MasterCmd()
            for command in (
                    'set automatic_html_opening False --no_save',
                    'set low_mem_multicore_nlo_generation False --no_save',
                    'set apply_flavor_grouping False --no_save',
                    'import model sm',
                    'generate u u~ > t t~ QED^2=0 QCD^2=4 [LOonly=QCD] --ewsudakov',
                    'add process u~ u > t t~ QED^2=0 QCD^2=4 [LOonly=QCD] --ewsudakov',
                    'output ewsudakovsa %s -f -nojpeg' % output):
                interface.exec_cmd(command, errorhandling=False, printcmd=False,
                                   precmd=True, postcmd=True)
            subprocesses = sorted((output/'SubProcesses').glob('P*'))
            self.assertEqual(len(subprocesses), 2)
            for sub in subprocesses:
                with self.subTest(subprocess=sub.name):
                    self.assertTrue((sub/'born.f').is_file())
                    # This standalone export has no native FKS tables. Both
                    # Born sources and Sudakov object dependencies must work
                    # without FKS runtime initialization or an f2py install.
                    result = subprocess.run(
                        ['make', 'born.o', 'ewsudakov_functions.o',
                         'ewsudakov_wrapper.o'], cwd=sub,
                        capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0,
                                     result.stdout + result.stderr)


class TestBornRegistry(unittest.TestCase):
    @unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
    def test_physical_native_real_router_preserves_flavour_and_model_state(self):
        from madgraph.iolibs import native_histories

        up = physical((2, 21, 23, 2))
        down = physical((1, 21, 23, 1))
        outer = dict(context=1, nincoming=2, sectors=[
            dict(sector=1, real=1, processes=[up]),
            dict(sector=2, real=1, processes=[down])])
        native = dict(context=2, nincoming=2, sectors=[
            dict(sector=1, real=1, processes=[down])])
        keys = [(1, 1), (1, 2), (2, 1)]
        # Rendering the borrowed PDF dispatcher needs its source. The real
        # router test does not link or execute the PDF dispatcher.
        sources = {2: {'parton_lum_1.f': (
            '      subroutine dlum_1(ans)\n'
            '      double precision ans\n'
            '      ans=0d0\n'
            '      end\n')}}

        with tempfile.TemporaryDirectory(prefix='mg7_native_router_') as tmp:
            root = Path(tmp)
            support.write_fortran(root/'nexternal.inc', [
                'integer,parameter::nexternal=4'])
            support.write_fortran(root/'orders.inc', [
                'integer,parameter::amp_split_size=2',
                'real(8) amp_split(amp_split_size)',
                'common/to_amp_split/amp_split'])
            support.write_fortran(root/'real_matrix_weights.inc', [
                'real(8) real_amp2(2)',
                'common/c_real_diagram_weights/real_amp2',
                'logical get_real_diagram_weights',
                'common/c_get_real_diagram_weights/get_real_diagram_weights'])
            support.write_fortran(root/'fks_info.inc', [
                'integer real_flavor_index_d(3)',
                'data real_flavor_index_d/1,2,99/'])

            # Supply the original explicit-buffer API so write_dispatchers
            # takes the production preservation path. No SMATRIX1 fallback
            # exists: linking would fail if native routing bypassed this API.
            # The stub deliberately leaves WGT_ME_REAL untouched.
            support.write_fortran(root/'real_me_chooser.f', [
                'subroutine smatrix_real(p,ret_amp_split,wgt)',
                'implicit none', "include 'nexternal.inc'",
                "include 'orders.inc'", "include 'fks_info.inc'",
                'real(8) p(0:3,nexternal),ret_amp_split(amp_split_size),wgt',
                'real(8) model_value', 'common/test_model/model_value',
                'integer nfksprocess,calls,last_selected',
                'common/c_nfksprocess/nfksprocess',
                'common/test_calls/calls',
                'common/test_selected/last_selected',
                'calls=calls+1', 'last_selected=nfksprocess',
                'wgt=100d0*real_flavor_index_d(nfksprocess)+p(0,1)+10d0*model_value',
                'ret_amp_split=[wgt,2d0*wgt]', 'end',
                'logical function mg7_nlo_real_is_fortran()',
                'implicit none', 'mg7_nlo_real_is_fortran=.false.', 'end'])
            native_histories.write_dispatchers(
                root, outer, {1: outer, 2: native}, keys, sources)
            self.assertTrue((root/'native_real_me_chooser.f').is_file())
            self.assertRegex((root/'real_me_chooser.f').read_text(),
                             r'(?i)subroutine\s+smatrix_real_local\s*\(')

            trace_modules = write_real_cache_fixture(root,4,2)
            program = [
                'program check', 'use mc_native_context', 'implicit none',
                "include 'orders.inc'", "include 'real_matrix_weights.inc'",
                'integer nfksprocess,calls,last_selected',
                'common/c_nfksprocess/nfksprocess',
                'common/test_calls/calls',
                'common/test_selected/last_selected',
                'real(8) model_value', 'common/test_model/model_value',
                'real(8) p(0:3,4),returned(amp_split_size),wgt',
                'real(8) wgt_me_born,wgt_me_real',
                'common/c_wgt_me_tree/wgt_me_born,wgt_me_real',
                'logical mg7_nlo_real_is_fortran',
                'external mg7_nlo_real_is_fortran',
                'if(mg7_nlo_real_is_fortran())stop 1',
                'calls=0', 'last_selected=0', 'model_value=1d0',
                'p=1d0', 'get_real_diagram_weights=.false.']

            def step(alias, expected, expected_calls, code):
                program.extend([
                    'nfksprocess=%d' % alias,
                    'wgt_me_real=-999d0', 'amp_split=-999d0',
                    'returned=-999d0', 'wgt=-999d0',
                    'call smatrix_real(p,returned,wgt)',
                    'if(wgt.ne.%dd0.or.calls.ne.%d)stop %d' %
                    (expected, expected_calls, code),
                    'if(nfksprocess.ne.%d)stop %d' % (alias, code+1),
                    'if(wgt_me_real.ne.wgt)stop %d' % (code+2),
                    'if(any(returned.ne.[wgt,2d0*wgt]))stop %d' % (code+3),
                    'if(any(amp_split.ne.returned))stop %d' % (code+4)])

            # Distinct physical aliases share the same numerical evaluator.
            # A -> B -> A must evaluate twice, then restore A from its cache.
            step(1, 111, 1, 10)
            step(2, 211, 2, 20)
            step(1, 111, 2, 30)
            # Borrowed alias 3 has raw flavour index 99. It must evaluate via
            # outer sector 2, then restore alias 3 on both miss and cache hit.
            step(3, 211, 3, 40)
            program.append('if(last_selected.ne.2)stop 45')
            step(3, 211, 3, 50)
            # Preserve the pre-existing model and momentum cache keys too.
            program.append('model_value=2d0')
            step(3, 221, 4, 60)
            program.append('shared_real_point(0,1)=2d0')
            step(3, 222, 5, 70)
            program.extend([
                'end',
                'subroutine mc_capture_model_state(state)',
                'use mc_born_types', 'implicit none',
                'type(BornModelState) state',
                'real(8) model_value', 'common/test_model/model_value',
                'state%configuration_epoch=1',
                'state%real_values=[model_value]',
                'state%complex_values=[(0d0,0d0)]', 'end'])
            support.write_fortran(root/'check.f', program)
            types = (Path(support.__file__).resolve().parents[2]/
                     'Template/NLO/Source/BornSupport/mc_born_types.f90')
            TestBornLibrary.run_command([
                'gfortran', '-fcheck=all', '-ffixed-line-length-none', '-ffunction-sections',
                '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections',
                '-ffree-line-length-none', str(types), *trace_modules, 'mc_native_context.f90',
                'real_me_chooser.f', 'native_real_me_chooser.f', 'check.f',
                '-o', 'check'], tmp)
            TestBornLibrary.run_command(['./check'], tmp)

    def test_readonly_aloha_records_preserve_the_dependency_proof(self):
        module = ['module aloha_object', 'type aloha', 'sequence',
                  'double complex::w(4)', 'double precision::p(0:3)',
                  'integer::flv_index', 'end type aloha', 'end module aloha_object']
        helper = ['subroutine helper(w)', 'use aloha_object', 'implicit none',
                  'type(aloha) w', 'w%w(1)=2d0*w%p(0)', 'end']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root/'Source/DHELAS'
            directory.mkdir(parents=True)
            path = directory/'aloha_functions.f'
            support.write_fortran(path, module+helper)
            self.assertTrue(support.explicit_helas_dependencies(root))
            # Module-level numeric state would be an input outside the graph.
            support.write_fortran(path, module[:-1]+['double precision hidden']+
                                  module[-1:]+helper)
            self.assertFalse(support.explicit_helas_dependencies(root))
            support.write_fortran(path, module+[line.replace('aloha_object', 'hidden_state')
                                               for line in helper])
            self.assertFalse(support.explicit_helas_dependencies(root))

    def test_quadruple_precision_aloha_records_preserve_the_dependency_proof(self):
        # Outputs with MadLoop virtuals install aloha_functions_loop.f, which
        # also defines MP_ALOHA/MP_ALOHA2D. Rejecting them dropped the
        # G-rescaling helicity predicate, so every tree helicity mask was
        # relearned whenever G changed (each point with a dynamic scale).
        module = ['module aloha_object']
        for name, size in (('aloha', 4), ('aloha2d', 16)):
            module += ['type %s' % name, 'sequence', 'double complex::w(%d)' % size,
                       'double complex :: p(0:3)', 'integer :: flv_index', 'end type %s' % name]
        for name, size in (('mp_aloha', 4), ('mp_aloha2d', 16)):
            module += ['type %s' % name, 'sequence', 'complex*32 :: w(%d)' % size,
                       'complex*32 :: p(0:3)', 'integer :: flv_index', 'end type %s' % name]
        module += ['end module aloha_object']
        helper = ['subroutine helper(w)', 'use aloha_object', 'implicit none',
                  'type(aloha) w', 'w%w(1)=2d0*w%p(0)', 'end',
                  'subroutine mp_helper(w)', 'use aloha_object', 'implicit none',
                  'type(mp_aloha) w', 'w%w(1)=2*w%p(0)', 'end']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root/'Source/DHELAS'
            directory.mkdir(parents=True)
            path = directory/'aloha_functions.f'
            support.write_fortran(path, module+helper)
            self.assertTrue(support.explicit_helas_dependencies(root))
            # Module-level state is still rejected inside an MP record module,
            # including declarations spelled like the record members.
            for hidden in ('complex*32 hidden', 'complex*32 :: hidden',
                           'real*16 :: hidden(2)'):
                support.write_fortran(path, module[:-1]+[hidden]+
                                      module[-1:]+helper)
                self.assertFalse(support.explicit_helas_dependencies(root), hidden)
            # Only the known record names are certified.
            support.write_fortran(path, [line.replace('mp_aloha2d', 'mp_other')
                                         for line in module]+helper)
            self.assertFalse(support.explicit_helas_dependencies(root))

    def test_flavour_coupling_records_preserve_the_dependency_proof(self):
        # ALOHA_OBJECT uses MODEL_OBJECT (flavour-coupling record types) and
        # contains the flavour-coupling accessors. Rejecting them dropped the
        # G-rescaling helicity predicate for every output, so each real
        # emission relearned its helicity mask whenever G changed.
        model_object = ['module model_object', 'type coupptr  ! pointer array',
                        'sequence', 'double complex, pointer :: p', 'end type coupptr',
                        'type flv_coupling', 'sequence', 'integer :: partner(0)',
                        'integer :: partner2(0)', 'type(coupptr) :: val(0)',
                        'end type flv_coupling', 'end module model_object',
                        'subroutine init_flv_couplings()', 'use model_object', 'end']
        module = ['module aloha_object', 'use model_object', 'type aloha', 'sequence',
                  'double complex::w(4)', 'double complex :: p(0:3)',
                  'integer :: flv_index', 'end type aloha', 'contains',
                  'double complex function get_flv_coupling_value(mcoup,k1,k2)',
                  'type(flv_coupling), intent(in) :: mcoup',
                  'integer, intent(in) :: k1,k2',
                  'get_flv_coupling_value=mcoup%val(k1)%p',
                  'end function get_flv_coupling_value', 'end module aloha_object']
        helper = ['subroutine helper(w)', 'use aloha_object', 'implicit none',
                  'type(aloha) w', 'w%w(1)=2d0*w%p(0)', 'end']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'Source/DHELAS').mkdir(parents=True)
            (root/'Source/MODEL').mkdir(parents=True)
            path = root/'Source/DHELAS/aloha_functions.f'
            model = root/'Source/MODEL/flavor_couplings.f'
            support.write_fortran(path, module+helper)
            # Without the model module the record types are unproven.
            self.assertFalse(support.explicit_helas_dependencies(root))
            support.write_fortran(model, model_object)
            self.assertTrue(support.model_object_holds_only_types(root))
            self.assertTrue(support.explicit_helas_dependencies(root))
            # Module-level state in MODEL_OBJECT is a hidden input.
            for hidden in ('double complex, target :: gc_1', 'double precision scale'):
                support.write_fortran(model, model_object[:11]+[hidden]+model_object[11:])
                self.assertFalse(support.model_object_holds_only_types(root), hidden)
                self.assertFalse(support.explicit_helas_dependencies(root), hidden)
            support.write_fortran(model, model_object)
            # A helper reading a coupling through the accessor is not certified,
            # nor is one declaring a flavour-coupling record.
            for body in (['double complex function vertex(m)', 'use aloha_object',
                          'type(flv_coupling) m', 'vertex=get_flv_coupling_value(m,1,1)',
                          'end'],
                         ['subroutine vertex(w,c)', 'use aloha_object', 'type(aloha) w',
                          'double complex c', 'c=get_flv_coupling_value(w,1,1)', 'end']):
                support.write_fortran(path, module+helper+body)
                self.assertFalse(support.explicit_helas_dependencies(root), body)

    def test_target_couplings_are_transferred_as_numeric_model_state(self):
        lines = ['double complex, target :: gc_10,gc_11',
                 'double complex, target :: gal(2)',
                 'common/couplings/gc_10,gc_11,gal']
        members, sizes = support.common_members(lines)
        self.assertEqual(members, [('gc_10', 'complex', '1'),
                                   ('gc_11', 'complex', '2'),
                                   ('gal', 'complex', '3:4')])
        self.assertEqual(sizes, dict(real=0, complex=4))

    def test_external_width_particles_from_born_metadata(self):
        """Width removal must survive the replacement of static Born tables."""
        with tempfile.TemporaryDirectory(prefix='mg5_born_pids_') as tmp:
            root = Path(tmp)
            sub = root/'SubProcesses'/'P0_epem_ttx'
            sub.mkdir(parents=True)
            (root/'SubProcesses'/'subproc.mg').write_text(sub.name+'\n')
            table = '      DATA (IDUP(I,1),I=1,4)/ -11, 11, 6, -6 /\n'
            (sub/'born_leshouche.inc').write_text(table)
            run = mock.Mock(me_dir=tmp)
            expected = {'-11', '11', '6', '-6'}
            self.assertEqual(CommonRunCmd.get_pid_final_initial_states(run), expected)
            (sub/'born_leshouche.inc').write_text(
                '      common/mc_native_born_lh/idup,mothup,icolup\n')
            (sub/'born_support.json').write_text(json.dumps(
                dict(context=1, born_pdgs=[[-11,11,6,-6],[-13,13,6,-6]])))
            self.assertEqual(CommonRunCmd.get_pid_final_initial_states(run),
                             expected | {'-13', '13'})
            # Outputs generated before born_pdgs was recorded remain usable.
            metadata = root/'Source'/'BornSupport'/'c1'
            metadata.mkdir(parents=True)
            (metadata/'metadata.f').write_text(table)
            (sub/'born_support.json').write_text(json.dumps(dict(context=1)))
            self.assertEqual(CommonRunCmd.get_pid_final_initial_states(run), expected)

    def test_foreign_flavours_and_group_order(self):
        u, d = physical((2,21,23,2)), physical((1,21,23,1))
        a = context(1,[u,d],4,1,[(4,1),(4,2)])
        b = context(2,[d,u],4,2,[(4,1),(4,2)],orders=('QED','QCD'))
        histories = support.resolve_histories([a,b])['P1'][0]
        self.assertEqual(histories['unresolved'], [])
        self.assertEqual(len(histories['histories']), 2)
        foreign = histories['histories'][1]
        self.assertEqual(foreign['flavours'], [2,1])
        self.assertEqual(foreign['orders'], [2,1])

    def test_labelled_identical_orientations(self):
        a = context(1,[physical((2,-2,21,21,21))],5,3,
                    [(i,j) for i in (3,4,5) for j in (3,4,5) if i != j])
        rows = support.resolve_histories([a])['P1'][0]['histories']
        self.assertEqual({(r['i'],r['j']) for r in rows},
                         {(i,j) for i in (3,4,5) for j in (3,4,5) if i != j})
        self.assertEqual(len(rows),6)

    def test_incoming_order_and_tags_preserved(self):
        native = physical((2,21,22,22),[False,False,False,True])
        outer = physical((2,21,22,22),[False,False,True,False])
        self.assertEqual(list(support.labelled_maps(native,outer,2)), [(1,2,4,3)])
        self.assertEqual(list(support.labelled_maps(native,
            physical((21,2,22,22)),2)), [])

    def test_identical_spectators_do_not_expand_factorially(self):
        physical_state = physical((2,-2)+(21,)*10)
        pairs = {(i,j) for i in range(3,13) for j in range(3,13) if i != j}
        a = context(1,[physical_state],12,11,sorted(pairs))
        rows = support.resolve_histories([a])['P1'][0]['histories']
        self.assertEqual(len(rows),90)
        self.assertEqual({(r['i'],r['j']) for r in rows},pairs)
        for row in rows:
            self.assertEqual(sorted(row['permutation']),list(range(1,13)))

    def test_unique_ownership(self):
        a = context(1,[physical((2,21,23,2))],4,1,[(4,1)])
        b = copy.deepcopy(a)
        b.update(context=2,provider=2,directory='P2')
        with self.assertRaisesRegex(MadGraph5Error,'Ambiguous global MC history'):
            support.resolve_histories([a,b])

    def test_missing_is_explicit(self):
        a = context(1,[physical((2,21,23,2))],4,1,[(4,1),(4,2)])
        self.assertEqual(support.resolve_histories([a])['P1'][0]['unresolved'], [(1,4,2)])

    def test_worker_order_does_not_change_registry(self):
        a = context(1,[physical((2,21,23,2))],4,1,[(4,1),(4,2)])
        b = context(2,[physical((2,21,23,2))],4,2,[(4,1),(4,2)])
        self.assertEqual(support.resolve_histories([a,b]),support.resolve_histories([b,a]))

    def test_native_provenance_round_trip(self):
        from madgraph.various.lhe_parser import OneNLOWeight
        # Five legs, native incoming flavours, coupling powers, scales,
        # momentum references, contribution type and native FKS identity.
        line = ('1 0 0 2 3 5 21 2 -11 12 1 404 2 0.1 0.2 '
                '100 100 100 1.2 1 2 13 123 5 1 2 4 1')
        legacy = OneNLOWeight(line)
        self.assertIsNone(legacy.native_provenance)
        native = OneNLOWeight(line+' 3 7 105 2')
        self.assertEqual(native.native_provenance,(3,7,105,2))
        restored = OneNLOWeight(native.__str__(mode='formatted'))
        self.assertEqual(restored.native_provenance,native.native_provenance)
        self.assertEqual(restored.pdgs,native.pdgs)
        self.assertEqual(restored.nfks,123)

    def test_namespacing_preserves_components_and_strings(self):
        text = "call born(p,result%born) ! 'born'"
        self.assertEqual(support.rename(text,{'born':'mcb1_born'}),
                         "call mcb1_born(p,result%born) ! 'born'")

    @unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
    def test_extra_helicity_filter_tracks_sector_and_model(self):
        # The extra-counterterm template has one NTRY/SKIP, unlike the main
        # Born's per-sector arrays. Its mother may change between sectors.
        routine = ['subroutine extra_filter(skip_out,tries)', 'implicit none',
            'integer skip_out,tries,ntry,nfksprocess,skip,glu_ij',
            'integer nhel(2,4),ij_values(2)',
            'common/c_nfksprocess/nfksprocess', 'logical goodhel(4)',
            'save ntry,skip,goodhel', 'data ntry/0/',
            'data nhel/-1,-1,1,-1,-1,1,1,1/', 'data ij_values/1,2/',
            'glu_ij=ij_values(nfksprocess)', 'NTRY=NTRY+1',
            'if(ntry.lt.2)then', 'skip=1',
            'do while(nhel(glu_ij,skip).ne.1)', 'skip=skip+1', 'enddo',
            'skip=skip-1', 'endif', 'skip_out=skip', 'tries=ntry', 'end']
        program = ['program check', 'implicit none',
            'integer nfksprocess,born_model_epoch,skip,tries',
            'common/c_nfksprocess/nfksprocess',
            'common/c_born_model_epoch/born_model_epoch',
            'born_model_epoch=1', 'nfksprocess=1', 'call extra_filter(skip,tries)',
            'if(skip.ne.1.or.tries.ne.1)stop 1', 'call extra_filter(skip,tries)',
            'if(skip.ne.1.or.tries.ne.2)stop 2',
            'nfksprocess=2', 'call extra_filter(skip,tries)',
            'if(skip.ne.2.or.tries.ne.1)stop 3',
            'born_model_epoch=2', 'call extra_filter(skip,tries)',
            'if(skip.ne.2.or.tries.ne.1)stop 4', 'end']
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'check.f'
            support.write_fortran(source,program+support.model_aware_helicity_filter(routine))
            TestBornLibrary.run_command(['gfortran','-fcheck=all','-ffixed-line-length-none',
                                         'check.f','-o','check'],tmp)
            TestBornLibrary.run_command(['./check'],tmp)

    @unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
    def test_born_zero_mask_relearns_when_only_the_frame_changes(self):
        routine = ['subroutine sum_born(p,ans,calls)', 'implicit none',
            'real(8) p,ans,t', 'integer calls,ntry,i,nfksprocess',
            'common/c_nfksprocess/nfksprocess', 'logical goodhel(2)',
            'save ntry,goodhel', 'data ntry/0/', 'data goodhel/2*.false./',
            'ntry=ntry+1', 'ans=0d0', 'calls=0', 'do i=1,2',
            'if(goodhel(i).or.ntry.lt.2)then', 'calls=calls+1',
            't=1d0', 'if(i.eq.2)t=p', 'ans=ans+t',
            'if(t.ne.0d0)goodhel(i)=.true.', 'endif', 'enddo', 'end']
        program = ['program check', 'implicit none',
            'integer nfksprocess,born_model_epoch,born_helicity_frame(2),calls',
            'common/c_nfksprocess/nfksprocess',
            'common/c_born_model_epoch/born_model_epoch',
            'common/c_born_helicity_frame/born_helicity_frame', 'real(8) ans',
            'nfksprocess=1', 'born_model_epoch=11', 'born_helicity_frame=0',
            'call sum_born(0d0,ans,calls)', 'if(ans.ne.1d0.or.calls.ne.2)stop 1',
            'call sum_born(0d0,ans,calls)', 'if(ans.ne.1d0.or.calls.ne.1)stop 2',
            'born_helicity_frame=[3,8]', 'call sum_born(2d0,ans,calls)',
            'if(ans.ne.3d0.or.calls.ne.2)stop 3',
            'call sum_born(2d0,ans,calls)', 'if(ans.ne.3d0.or.calls.ne.2)stop 4',
            # The same run-card mask can identify another basis in a reused
            # provider when the native context's user-leg mapping changes.
            'born_helicity_frame=[4,8]', 'call sum_born(0d0,ans,calls)',
            'if(ans.ne.1d0.or.calls.ne.2)stop 5',
            'call sum_born(0d0,ans,calls)', 'if(ans.ne.1d0.or.calls.ne.1)stop 6',
            'born_helicity_frame=0', 'call sum_born(0d0,ans,calls)',
            'if(ans.ne.1d0.or.calls.ne.2)stop 7',
            'if(born_model_epoch.ne.11)stop 8', 'end']
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'check.f'
            support.write_fortran(source, program+support.model_aware_helicity_filter(routine))
            TestBornLibrary.run_command(['gfortran','-fcheck=all','-ffixed-line-length-none',
                                         'check.f','-o','check'],tmp)
            TestBornLibrary.run_command(['./check'],tmp)

    @unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
    def test_born_completed_sectors_relearn_after_model_and_frame_changes(self):
        # Like the scalar Born, the dispatcher owns DO_TRY while its learning
        # subroutine owns the per-sector counts and the shared helicity mask.
        routines = """subroutine evaluate(nfksprocess,expected,weight,calls)
implicit none
integer nfksprocess,expected,weight,calls,i
logical do_try(2),goodhel(2,2)
data do_try/2*.true./
common/c_goodhel/goodhel
weight=0
calls=0
if(do_try(nfksprocess)) then
call learn(nfksprocess,do_try(nfksprocess),expected,weight,calls)
else
do i=1,2
if(.not.goodhel(i,nfksprocess))cycle
calls=calls+1
if(i.eq.expected)weight=weight+1
enddo
endif
end
subroutine learn(nfksprocess,learning,expected,weight,calls)
implicit none
integer nfksprocess,expected,weight,calls,ntry(2),i
logical learning,goodhel(2,2)
data ntry/2*0/
common/c_goodhel/goodhel
ntry(nfksprocess)=ntry(nfksprocess)+1
do i=1,2
if(.not.goodhel(i,nfksprocess).and.ntry(nfksprocess).ge.2)cycle
calls=calls+1
if(i.eq.expected)then
weight=weight+1
goodhel(i,nfksprocess)=.true.
endif
enddo
if(ntry(nfksprocess).ge.2)learning=.false.
end""".splitlines()
        program = """program check
implicit none
integer epoch,frame(2)
common/c_born_model_epoch/epoch
common/c_born_helicity_frame/frame
epoch=1
frame=0
call check_sector(1,1,2)
call check_sector(1,1,1)
call check_sector(1,1,1)
epoch=2
call check_sector(2,2,2)
call check_sector(1,2,2)
call check_sector(1,2,1)
call check_sector(2,2,1)
frame=[7,3]
call check_sector(2,1,2)
call check_sector(1,1,2)
call check_sector(1,1,1)
contains
subroutine check_sector(sector,expected,count)
integer sector,expected,count,weight,calls
call evaluate(sector,expected,weight,calls)
if(weight.ne.1.or.calls.ne.count)stop 1
end subroutine
end""".splitlines()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'check.f'
            support.write_fortran(source, program+support.model_aware_helicity_filter(routines))
            TestBornLibrary.run_command(['gfortran','-fcheck=all','-ffixed-line-length-none',
                                         'check.f','-o','check'],tmp)
            TestBornLibrary.run_command(['./check'],tmp)

    @unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
    def test_cached_helicities_preserve_the_other_frame_zero_mask(self):
        # Compile the production adapter and mask transformation together.
        # The two bases have disjoint nonzero amplitudes, so an intervening
        # learned mask cannot accidentally give the right optional weights.
        record = dict(provider=1,nexternal=4,ngraphs=1,ncolor=1,nhelicity=2,
            namps=1,nsqamps=1,orders=['QCD'],links=[],
            helicity_matrices=[{}],sectors=[dict(fks=dict(splitting_type=['QCD']))])
        declarations = ["include 'orders.inc'", 'real(8) p(0:3,3),ans',
            'complex(8) saveamp(1,2)', 'common/to_saveamp/saveamp',
            'logical goodhel(2,1)', 'common/c_goodhel/goodhel',
            'integer nfksprocess', 'common/c_nfksprocess/nfksprocess']
        routine = ['subroutine sborn(p,ans)', 'implicit none']+declarations+[
            'integer ntry,i,calls', 'common/test_calls/calls',
            'logical calculatedBorn', 'common/ccalculatedBorn/calculatedBorn',
            'real(8) amp2(1),jamp2(0:1),amp2b(1),jamp2b(0:1,0:1)',
            'common/to_amps/amp2,jamp2', 'common/to_amps_born/amp2b,jamp2b',
            'save ntry', 'data ntry/0/', 'ntry=ntry+1',
            'if(.not.calculatedBorn)then', 'saveamp=0d0', 'do i=1,2',
            'if(goodhel(i,1).or.ntry.lt.2)then', 'calls=calls+1',
            'saveamp(1,i)=p(0,i)',
            'if(saveamp(1,i).ne.(0d0,0d0))goodhel(i,1)=.true.',
            'endif', 'enddo', 'endif', 'ans=sum(abs(saveamp)**2)',
            'amp_split=ans', 'amp2=ans', 'amp2b=ans', 'jamp2=ans', 'jamp2b=ans', 'end']
        optional = ['subroutine sborn_hel(p,ans)', 'implicit none']+declarations+[
            'integer i', 'real(8) hel(2),helorders(1,2)',
            'common/c_born_hel/hel', 'common/c_born_hel_split/helorders',
            'do i=1,2', 'if(goodhel(i,1))hel(i)=abs(saveamp(1,i))**2', 'enddo',
            'helorders(1,:)=hel', 'ans=sum(hel)', 'end',
            'subroutine sborn_sf(p,m,n,ans)', 'real(8) p(0:3,3),ans',
            'integer m,n', 'stop 90', 'end',
            'subroutine extra_cnt(p,n,ans)', 'real(8) p(0:3,3)',
            'integer n', 'complex(8) ans(2,1)', 'stop 91', 'end',
            'subroutine sborn_onehel(p,nhel,ihel,ans)', 'real(8) p(0:3,3),ans',
            'integer nhel(3),ihel', 'stop 92', 'end']
        program = ['program check', 'use mc_born_types', 'implicit none',
            'type(BornModelState) state', 'type(BornRequest) request',
            'type(BornResult) result', 'real(8) p(0:3,3)', 'integer status,calls',
            'common/test_calls/calls', 'calls=0',
            'allocate(state%real_values(0),state%complex_values(0),request%charges(3))',
            'request%charges=0d0', 'p=0d0', 'p(0,1)=1d0',
            'call mcb_provider_1(1,p,state,request,result,status)',
            'if(status.ne.0.or.result%born.ne.1d0.or.calls.ne.2)stop 1',
            # Learn B's disjoint mask, then return to the cached point A with
            # optional helicities. Neither HELAS nor the B mask may change.
            'p=0d0', 'p(0,2)=2d0', 'request%helicity_frame=[1,4]',
            'call mcb_provider_1(1,p,state,request,result,status)',
            'if(status.ne.0.or.result%born.ne.4d0.or.calls.ne.4)stop 2',
            'p=0d0', 'p(0,1)=1d0', 'request%helicity_frame=0',
            'request%helicities=.true.',
            'call mcb_provider_1(1,p,state,request,result,status)',
            'if(status.ne.0.or.calls.ne.4)stop 3',
            'if(any(result%helicities.ne.[1d0,0d0]))stop 4',
            'if(any(result%helicity_orders(1,:).ne.[1d0,0d0]))stop 5',
            # An uncached point in B still uses its correctly learned mask.
            # This detects an epoch changed to A while restoring B's mask.
            'p=0d0', 'p(0,2)=4d0', 'request%helicity_frame=[1,4]',
            'call mcb_provider_1(1,p,state,request,result,status)',
            'if(status.ne.0.or.result%born.ne.16d0.or.calls.ne.5)stop 6',
            'if(any(result%helicities.ne.[0d0,16d0]))stop 7',
            # A new A point must relearn both columns. The unfiltered oracle
            # is simply the squared norm of these two supplied amplitudes.
            'p=0d0', 'p(0,1)=3d0', 'request%helicity_frame=0',
            'call mcb_provider_1(1,p,state,request,result,status)',
            'if(status.ne.0.or.result%born.ne.sum(p(0,1:2)**2))stop 8',
            'if(calls.ne.7.or.any(result%helicities.ne.[9d0,0d0]))stop 9', 'end']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'SubProcesses/P0'
            source.mkdir(parents=True)
            (root/'Source/MODEL').mkdir(parents=True)
            (source/'coupl.inc').touch()
            (root/'Source/MODEL/input.inc').touch()
            support.write_fortran(source/'orders.inc',[
                'integer,parameter::amp_split_size=1,nsplitorders=1',
                'real(8) amp_split(1)', 'common/to_amp_split/amp_split',
                'complex(8) amp_split_cnt(1,2,1)', 'common/to_amp_split_cnt/amp_split_cnt'])
            support.write_fortran(source/'check.f',program+
                support.provider_adapter(record,source,[])+
                support.model_aware_helicity_filter(routine)+optional)
            types = Path(support.__file__).resolve().parents[2]/'Template/NLO/Source/BornSupport/mc_born_types.f90'
            TestBornLibrary.run_command(['gfortran','-fcheck=all','-ffixed-line-length-none',
                str(types),'check.f','-o','check'],source)
            TestBornLibrary.run_command(['./check'],source)

    @unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
    def test_polarized_real_cache_preserves_caller_frame(self):
        from madgraph.iolibs import native_histories
        identity = physical((21,21,6))
        restricted = copy.deepcopy(identity)
        restricted = (restricted[0],restricted[1][:-1]+((6,True,(1,),False,None),))
        record = dict(context=1,sectors=[dict(real=1,processes=[restricted]),
                                        dict(real=2,processes=[identity])])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support.write_fortran(root/'nexternal.inc', ['integer,parameter::nexternal=3'])
            support.write_fortran(root/'orders.inc', ['integer,parameter::amp_split_size=1',
                'real(8) amp_split(1)', 'common/to_amp_split/amp_split'])
            support.write_fortran(root/'real_matrix_weights.inc', [
                'real(8) real_amp2(2)', 'common/c_real_diagram_weights/real_amp2',
                'logical get_real_diagram_weights',
                'common/c_get_real_diagram_weights/get_real_diagram_weights'])
            support.write_fortran(root/'fks_info.inc', [
                'integer,parameter::real_flavor_index_d(2)=[1,1]'])
            native_histories.write_dispatchers(root,record,{1:record},[(1,1),(1,2)],{})
            trace_modules = write_real_cache_fixture(root,3,1)
            program = ['program check', 'use mc_native_context', 'implicit none',
                "include 'real_matrix_weights.inc'",
                'integer nfksprocess,calls', 'common/c_nfksprocess/nfksprocess',
                'common/test_calls/calls', 'real(8) p(0:3,3),wgt',
                'calls=0', 'get_real_diagram_weights=.false.',
                'shared_real_point=1d0', 'p=2d0', 'nfksprocess=1',
                'call smatrix_real(p,wgt)', 'if(wgt.ne.4d0.or.calls.ne.1)stop 1',
                'p(0,1)=3d0', 'call smatrix_real(p,wgt)',
                'if(wgt.ne.5d0.or.calls.ne.2)stop 2', 'call smatrix_real(p,wgt)',
                'if(wgt.ne.5d0.or.calls.ne.2)stop 3', 'nfksprocess=2',
                'call smatrix_real(p,wgt)', 'if(wgt.ne.2d0.or.calls.ne.3)stop 4',
                'p(0,1)=4d0', 'call smatrix_real(p,wgt)',
                'if(wgt.ne.2d0.or.calls.ne.3)stop 5',
                # A scalar cache hit cannot supply weights that were never
                # requested; a later weights request must evaluate once.
                'get_real_diagram_weights=.true.', 'call smatrix_real(p,wgt)',
                'if(calls.ne.4.or.any(real_amp2.ne.[2d0,4d0]))stop 6',
                'real_amp2=-1d0', 'call smatrix_real(p,wgt)',
                'if(calls.ne.4.or.any(real_amp2.ne.[2d0,4d0]))stop 7', 'end',
                'subroutine mc_capture_model_state(state)', 'use mc_born_types',
                'implicit none', 'type(BornModelState) state',
                'state%real_values=[1d0]', 'state%complex_values=[(1d0,0d0)]', 'end']
            for number in (1,2):
                program += ['subroutine smatrix%d(p,wgt)' % number, 'implicit none',
                    "include 'real_matrix_weights.inc'",
                    "include 'orders.inc'", 'real(8) p(0:3,3),wgt,wgt_me_born,wgt_me_real',
                    'integer calls', 'common/test_calls/calls',
                    'common/c_wgt_me_tree/wgt_me_born,wgt_me_real',
                    'calls=calls+1', 'wgt=p(0,1)+p(0,3)',
                    'amp_split=wgt', 'wgt_me_real=wgt',
                    'if(get_real_diagram_weights)real_amp2=[wgt,2d0*wgt]', 'end']
            support.write_fortran(root/'check.f',program)
            types = Path(support.__file__).resolve().parents[2]/'Template/NLO/Source/BornSupport/mc_born_types.f90'
            TestBornLibrary.run_command(['gfortran','-fcheck=all','-ffixed-line-length-none',
                '-ffree-line-length-none',str(types),*trace_modules,'mc_native_context.f90',
                'real_me_chooser.f','check.f','-o','check'],tmp)
            TestBornLibrary.run_command(['./check'],tmp)


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestBornLibrary(unittest.TestCase):
    """Use actual exports and independently compiled pre-wrapper evaluators."""

    origin = '@loader_path' if sys.platform == 'darwin' else '$ORIGIN'
    selected_processes = None

    @classmethod
    def setUpClass(cls):
        from madgraph.interface import master_interface
        cls.temp = tempfile.TemporaryDirectory(prefix='mg5_born_support_test_')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.work = Path(cls.temp.name)
        cls.outputs = {}
        cls.cache_metrics = {}
        interface = master_interface.MasterCmd()
        def command(text):
            interface.exec_cmd(text,errorhandling=False,printcmd=False,
                               precmd=True,postcmd=True)
        command('set automatic_html_opening False --no_save')
        command('set low_mem_multicore_nlo_generation False --no_save')
        command('import model sm')
        command('define p = g u u~ d d~')
        command('define jj = g a')
        original_finalize = support.finalize

        def capture(exporter):
            root = Path(exporter.dir_path)
            reference = root/'reference'
            reference.mkdir()
            for sub in (root/'SubProcesses').glob('P*'):
                if not sub.is_dir():
                    continue
                target = reference/sub.name
                target.mkdir()
                for pattern in ('*.inc','born.f','born_hel.f','sborn_sf.f',
                                'b_sf_*.f','born_cnt_*.f','extra_cnt_wrapper.f',
                                'splitorders_stuff.f','ewsudakov_functions*.f'):
                    for source in sub.glob(pattern):
                        if source.exists():
                            shutil.copyfile(source,target/source.name)
            # A reused provider deliberately has no second standalone Born
            # implementation. Test its context against the representative's
            # original evaluator while retaining its own metadata includes.
            for name, entry in exporter.born_support_registry.items():
                if name == entry['source']:
                    continue
                for source in (reference/entry['source']).glob('*.f'):
                    target = reference/name/source.name
                    if not target.exists():
                        shutil.copyfile(source,target)
            return original_finalize(exporter)

        for name, process in (
                ('dy','p p > e+ e- QED^2=4 QCD^2=0 [real=QCD]'),
                ('ttbar','p p > t t~ QED^2=0 QCD^2=4 [real=QCD]'),
                ('wjet','p p > w+ j QED^2=2 QCD^2=2 [real=QCD]'),
                ('extra','u u~ > jj jj QED^2=4 QCD^2=4 [real=QCD]'),
                ('mixed','u u~ > d d~ QED^2=4 QCD^2=4 [real=QCD]'),
                ('gluons','g g > g g QED^2=0 QCD^2=4 [real=QCD]'),
                ('qed','e+ e- > mu+ mu- QED^2=4 QCD^2=0 [real=QED]'),
                ('loonly','u d~ > w+ g QED^2=2 QCD^2=2 [LOonly=QCD]'),
                ('shared',('u u~ > e+ e- QED^2=4 QCD^2=0 [real=QCD] @1',
                           'u u~ > e+ e- QED^2=4 QCD^2=0 [real=QCD] @2'))):
            if cls.selected_processes is not None and name not in cls.selected_processes:
                continue
            processes = (process,) if isinstance(process,str) else process
            command('generate '+processes[0])
            for additional in processes[1:]:
                command('add process '+additional)
            output = cls.work/name
            with mock.patch.object(support,'finalize',capture):
                command('output %s -f -nojpeg' % output)
            # The output dimension counts global squared-order slots. The
            # number of coupling names can be different for mixed amplitudes.
            registry = json.loads((output/'Source/BornSupport/registry.json').read_text())
            for rec in registry['contexts']:
                directory = output/'SubProcesses'/rec['directory']
                text = (directory/'mc_native_context.f90').read_text()
                nslot = int(re.search(r'provider_orders=(\d+)',text)[1])
                generated = (directory/'orders.inc').read_text()
                native = int(re.search(r'\bAMP_SPLIT_SIZE\s*=\s*(\d+)',generated,re.I)[1])
                if nslot != native:
                    raise AssertionError((rec['directory'],nslot,native))
            cls.instrument_provider_calls(output)
            cls.run_command(['make','-j2','FFLAGS=-O0 -g -fcheck=all -fbacktrace'],
                            output/'Source/BornSupport')
            cls.outputs[name] = output

    @staticmethod
    def instrument_provider_calls(output):
        """Count actual HELAS calls in test exports, without production counters."""
        directory = output/'Source/BornSupport'
        providers = sorted(directory.glob('p[0-9]*/matrix.f'))
        types = directory/'mc_born_types.f90'
        types.write_text(types.read_text().replace('  implicit none',
            '  implicit none\n  integer(8),save :: born_test_helas_calls(%d)=0\n'
            '  integer(8),save :: born_test_contractions(%d)=0' % (len(providers),len(providers)), 1))
        for path in providers:
            provider = int(path.parent.name[1:])
            lines = []
            for line in support.statements(path.read_text()):
                if re.match(r'(?i)\s*call\s+mcb\d+_sborn_splitorders\(', line):
                    lines.append('born_test_contractions(%d)=born_test_contractions(%d)+1' %
                                 (provider,provider))
                if re.match(r'(?i)\s*call\s+(?!mcb\d+_)\w+\s*\(', line):
                    lines.append('born_test_helas_calls(%d)=born_test_helas_calls(%d)+1' %
                                 (provider,provider))
                lines.append(line)
                if re.match(r'(?i)\s*(?:[\w*]+\s+)*?(?:subroutine|function)\s+\w+', line):
                    lines.append('use mc_born_types,only:born_test_helas_calls,born_test_contractions')
            support.write_fortran(path,lines)

    @staticmethod
    def run_command(command, cwd):
        result = subprocess.run(command,cwd=cwd,text=True,stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        if result.returncode:
            raise AssertionError(' '.join(map(str,command))+'\n'+result.stdout)
        return result.stdout

    def compare(self, output):
        registry = json.loads((output/'Source/BornSupport/registry.json').read_text())
        model_lines = support.expand(output/'Source/MODEL/coupl.inc')
        model_lines += support.expand(output/'Source/MODEL/input.inc')
        members,sizes = support.common_members(model_lines)
        layout = model_state_checks.load_layout(output)
        signature = json.loads((output/'Source/BornSupport/helicity_signature.json').read_text())
        self.assertIn(signature['mode'], ('homogeneous_G', 'exact_compact_state'))
        for record in registry['contexts']:
            with self.subTest(process=output.name, provider=record['provider']):
                source = output/'reference'/record['directory']
                # Extract only pure order helpers; their other routines depend
                # on event bookkeeping which is not part of this test.
                helpers = (source/'splitorders_stuff.f').read_text()
                helpers += '\n' + next(source.glob('ewsudakov_functions*.f')).read_text()
                support.write_fortran(source/'helpers.f',sum((support.routine(helpers,n)
                    for n in ('orders_to_amp_split_pos','amp_split_pos_to_orders',
                              'orders_equal','get_lo2_orders')),[]))
                helicity_data = [line for line in support.statements((source/'born.f').read_text())
                                 if re.match(r'(?i)data\s*\(nhel\(', line)]
                support.write_fortran(source/'check.f',self.driver(record,members,sizes,model_lines,
                    registry['contexts'], helicity_data, signature['signature'], layout))
                from tests.unit_tests.iolibs.born_reference import write_reference
                binding = output/'Source/MODEL/flavor_couplings.f'
                reference = write_reference(source, record, binding.is_file())
                paths = [binding] if binding.is_file() else []
                # HELAS is independent here; only the provider implementation
                # is linked dynamically, with hidden internal symbols.
                helas = output/'Source/DHELAS'
                aloha_module = helas/'aloha_functions.f'
                if aloha_module.is_file():
                    paths.append(aloha_module)
                paths += sorted(p for p in helas.glob('*.f') if p != aloha_module)
                paths += [output/'Source/MODEL/model_functions.f',reference,
                          source/'helpers.f',source/'check.f']
                self.run_command(['gfortran','-O0','-g','-fcheck=all','-ffixed-line-length-none',
                    '-ffree-line-length-none','-I'+str(output/'Source/MODEL'),
                    '-I'+str(source),'-I'+str(output/'lib'),
                    '-I'+str(output/'Source/BornSupport'),*map(str,paths),
                    '-L'+str(output/'lib'),'-lmc_born_support',
                    '-Wl,-rpath,'+self.origin+'/../../lib','-Wl,-rpath,'+self.origin,
                    '-o','check'],source)
                checked = self.run_command(['./check'],source)
                self.assertIn('PASS Born provider',checked)
                discovery,pruned = map(int,re.search(
                    r'HELAS discovery/pruned\s+(\d+)\s+(\d+)',checked).groups())
                self.cache_metrics[output.name,record['provider']] = (discovery,pruned)
                if output.name == 'dy':
                    self.assertLess(pruned,discovery,
                        'fixed-model helicity filtering must eliminate zero-helicity HELAS calls')
                # A worker only needs the executable and the DSO beside it.
                scratch = output/('scratch_'+str(record['context']))
                scratch.mkdir()
                shutil.copy2(source/'check',scratch/'check')
                for lib in (output/'lib').glob('libmc_born_support.*'):
                    shutil.copy2(lib,scratch/lib.name)
                self.assertIn('PASS Born provider',self.run_command(['./check'],scratch))

    @staticmethod
    def driver(r, members, sizes, model_lines, contexts, helicity_data=(), signature=None,
               layout=None):
        # This driver links the original Born sources captured before provider
        # finalization. Its calculatedBorn resets belong to that independent
        # reference evaluator, which still needs them to invalidate SAVEAMP.
        n,ng,nc,nh,ns = r['nexternal']-1,r['ngraphs'],r['ncolor'],r['nhelicity'],r['nsqamps']
        if layout:
            signature = model_state_checks.full_signature(signature, members, layout)
        lines = ['program check', 'use mc_born_support', 'use mc_born_types', 'implicit none']
        lines += ["include 'orders.inc'"]+model_lines+[
            'type(BornModelState) state,alternate', 'type(BornRequest) request',
            'type(BornResult) result,other', 'type(BornMetadata) metadata',
            'integer nfksprocess,status,sector,iteration,i,j,k,ihel',
            'integer(8) calls_before,calls_first,calls_pruned,contractions_before',
            'integer nhel(%d,%d)' % (n,nh),
            *helicity_data, 'common/c_nfksprocess/nfksprocess',
            'complex(8) ewsud(amp_split_size),ewsud_lo2(amp_split_size)',
            'common/to_amp_split_ewsud/ewsud',
            'common/to_amp_split_ewsud_lo2/ewsud_lo2',
            'logical calculatedBorn', 'common/ccalculatedBorn/calculatedBorn',
            'logical need_color_links,need_charge_links', 'common/c_need_links/need_color_links,need_charge_links',
            'logical split_type_used(nsplitorders)', 'common/to_split_type_used/split_type_used',
            'real(8) p(0:3,%d),pother(0:3,%d),ans,corr,mass,energy,magnitude' % (n,n),
            'real(8) amp2(%d),jamp2(0:%d)' % (ng,nc), 'common/to_amps/amp2,jamp2',
            'real(8) amp2b(%d),jamp2b(0:%d,0:%d)' % (ng,nc,r['namps']),
            'common/to_amps_born/amp2b,jamp2b',
            'complex(8) ans_cnt(2,nsplitorders),extra(2,nsplitorders)', 'common/c_born_cnt/ans_cnt',
            'real(8) hel(%d),helorders(%d,%d)' % (nh,ns,nh),
            'common/c_born_hel/hel', 'common/c_born_hel_split/helorders',
            'real(8) soft(amp_split_size)', 'common/to_amp_split_soft/soft',
            'real(8) charges(%d)' % n, 'common/c_charges_born/charges',
            'allocate(state%%real_values(%d),state%%complex_values(%d))' % (sizes['real'],sizes['complex']),
            'allocate(request%%charges(%d))' % n,
            'state%real_values=1.1d0', 'state%complex_values=(0.7d0,0.2d0)',
            'split_type_used=.true.', 'charges=0.5d0', 'request%charges=charges',
            'request%helicities=.true.',
            'call born_query(%d,%d,metadata,status)' % (r['provider'],r['context']),
            'if(status.ne.0)stop 10',
            'if(size(metadata%%born_ids,1).ne.%d)stop 14' % n,
            'if(size(metadata%%born_ids,2).ne.%d)stop 18' % r['nprocesses'],
            'if(any(metadata%sprop.ne.0.and.metadata%tprid.ne.0))stop 19',
            'if(lbound(metadata%configurations,1).ne.0)stop 24',
            'if(lbound(metadata%tree,2).ne.-size(metadata%tree,2))stop 25',
            'call born_query(%d,%d,metadata,status,require_complete=.true.)' % (r['provider'],r['context']),
            'if(all(metadata%complete_histories))then',
            'if(status.ne.0)stop 15', 'else', 'if(status.ne.5)stop 16', 'endif',
            'do iteration=1,3',
            'state%real_values=state%real_values*1.03d0',
            'state%complex_values=state%complex_values*(0.9d0,0.1d0)']
        lines += ['%s=state%%%s_values(%s)' % v for v in members]
        lines += ['mass=state%real_values(1)', 'energy=500d0',
            'magnitude=sqrt(energy**2-mass**2)', 'p=0d0',
            'p(:,1)=[energy,0d0,0d0,energy]', 'p(:,2)=[energy,0d0,0d0,-energy]',
            'p(:,3)=[energy,magnitude*0.6d0,0d0,magnitude*0.8d0]',
            'p(:,4)=[energy,-magnitude*0.6d0,0d0,-magnitude*0.8d0]',
            # Rotate transverse momenta while leaving the old cache's E,pz key
            # unchanged. This was not caught by the legacy savemom check.
            'if(mod(iteration,2).eq.0)then', 'p(2,:)=p(1,:)', 'p(1,:)=0d0','endif',
            'do sector=1,%d' % max(1,len(r['sectors'])),
            'nfksprocess=sector', 'request%sector=sector',
            'request%colour=.false.', 'request%charge=.false.', 'request%extra=0',
            'ans_cnt=(0d0,0d0)', 'amp_split_cnt=(0d0,0d0)',
            'calculatedBorn=.false.', 'call sborn(p,ans)',
            'call born_evaluate(%d,%d,p,state,request,result,status)' % (r['provider'],r['context']),
            'if(status.ne.0)stop 11',
            'calls_before=born_test_helas_calls(%d)' % r['provider'],
            'contractions_before=born_test_contractions(%d)' % r['provider'],
            'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
            'if(status.ne.0)stop 29',
            'if(born_test_helas_calls(%d).ne.calls_before)stop 30' % r['provider'],
            'if(born_test_contractions(%d).ne.contractions_before)stop 47' % r['provider'],
            'call close_real(result%born,other%born)',
            # A basis key invalidates only the learned zero mask. Identical
            # p/model values still identify the same complete numerical result.
            'request%helicities=.false.',
            'request%%helicity_frame=[%d,8]' % r['context'],
            'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
            'if(status.ne.0)stop 51', 'call close_real(result%born,other%born)',
            'if(born_test_helas_calls(%d).ne.calls_before)stop 52' % r['provider'],
            'if(born_test_contractions(%d).ne.contractions_before)stop 53' % r['provider'],
            'request%helicity_frame=0', 'request%helicities=.true.',
            'call close_real(ans,result%born)',
            'do i=1,amp_split_size', 'call close_real(amp_split(i),result%amplitudes(i))','enddo',
            'do i=1,%d' % ng, 'call close_real(amp2(i),result%diagrams(i))','enddo',
            'do i=1,%d' % nc, 'call close_real(jamp2(i),result%flows(i))','enddo',
            'do i=1,%d' % nc, 'do j=1,%d' % r['namps'],
            'call close_real(jamp2b(i,j),result%flow_orders(i,j))','enddo','enddo',
            'do j=1,nsplitorders','do k=1,2',
            'call close_complex(ans_cnt(k,j),result%counterterms(k,j))',
            'do i=1,amp_split_size',
            'call close_complex(amp_split_cnt(i,k,j),result%split_counterterms(i,k,j))',
            'enddo','enddo','enddo',
            'call sborn_hel(p,ans)',
            'do i=1,%d' % nh, 'call close_real(hel(i),result%helicities(i))','enddo',
            'do i=1,%d' % nh,'do j=1,%d' % ns,
            'call close_real(helorders(j,i),result%helicity_orders(j,i))','enddo','enddo',
            'do ihel=1,%d' % nh,
            'call sborn_onehel(p,nhel(:,ihel),ihel,ans)',
            'request%single_helicity=ihel', 'request%helicity=nhel(:,ihel)',
            'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
            'if(status.ne.0)stop 28', 'call close_real(ans,other%single_helicity)',
            'do i=1,amp_split_size',
            'call close_complex(ewsud(i),other%ewsudakov(i))',
            'call close_complex(ewsud_lo2(i),other%ewsudakov_lo2(i))',
            'enddo', 'enddo', 'request%single_helicity=0',
            'pother=p*1.13d0']
        for other in contexts:
            lines += ['request%sector=1',
                'call born_evaluate(%d,%d,pother,state,request,other,status)' % (other['provider'],other['context']),
                'if(status.ne.0)stop 12']
        lines += ['request%sector=sector',
            'calls_before=born_test_helas_calls(%d)' % r['provider'],
            'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
            'if(born_test_helas_calls(%d).ne.calls_before)stop 48' % r['provider'],
            'call close_real(result%born,other%born)']
        for m,n_ in r['links'] if r['sectors'] else []:
            lines += ['calculatedBorn=.false.', 'call sborn(p,ans)',
                'need_color_links=.true.', 'need_charge_links=.false.',
                'call sborn_sf(p,%d,%d,corr)' % (m,n_),
                'request%colour=.true.', 'request%m='+str(m), 'request%n='+str(n_),
                'calls_before=born_test_helas_calls(%d)' % r['provider'],
                'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
                'if(status.ne.0)stop 13', 'call close_real(corr,other%correlation)',
                'if(born_test_helas_calls(%d).ne.calls_before)stop 31' % r['provider'],
                'do i=1,amp_split_size','call close_real(soft(i),other%soft(i))','enddo']
        # QED uses the Born charge correlator even for a colourless process.
        # Interleave it with colour/Born calls to check the request flags and
        # private charge state, and compare all native split orders.
        if any('QED' in sector['fks']['splitting_type'] for sector in r['sectors']):
            lines += ['request%colour=.false.', 'request%charge=.true.',
                'need_color_links=.false.', 'need_charge_links=.true.',
                'do j=1,%d' % (n-1), 'do k=j+1,%d' % n,
                'calculatedBorn=.false.', 'call sborn(p,ans)',
                'call sborn_sf(p,j,k,corr)', 'request%m=j', 'request%n=k',
                'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
                'if(status.ne.0)stop 26', 'call close_real(corr,other%correlation)',
                'do i=1,amp_split_size','call close_real(soft(i),other%soft(i))','enddo',
                'charges=0.7d0', 'request%charges=charges',
                'call sborn_sf(p,j,k,corr)',
                'calls_before=born_test_helas_calls(%d)' % r['provider'],
                'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
                'if(status.ne.0)stop 40', 'call close_real(corr,other%correlation)',
                'if(born_test_helas_calls(%d).ne.calls_before)stop 41' % r['provider'],
                'charges=0.5d0', 'request%charges=charges',
                'enddo','enddo','request%charge=.false.', 'need_charge_links=.false.']
        if not r['sectors']:
            lines += ['request%charge=.true.', 'request%m=1', 'request%n=2',
                'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
                'if(status.ne.BORN_MISSING_CORRELATION)stop 27', 'request%charge=.false.']
        for sector in r['sectors']:
            icnt = sector['fks']['extra_cnt_index']+1
            if icnt:
                lines += ['if(sector.eq.%d)then' % sector['sector'],
                    'request%colour=.false.', 'request%%extra=%d' % icnt,
                    'call extra_cnt(p,%d,extra)' % icnt,
                    'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
                    'if(status.ne.0)stop 17',
                    'do j=1,nsplitorders','do k=1,2',
                    'call close_complex(extra(k,j),other%extra(k,j))',
                    'do i=1,amp_split_size',
                    'call close_complex(amp_split_cnt(i,k,j),other%split_counterterms(i,k,j))',
                    'enddo','enddo','enddo','endif']
        lines += ['enddo','enddo',
            'request%sector=1', 'request%colour=.false.', 'request%charge=.false.',
            'request%extra=0', 'request%helicities=.false.',
            'call born_evaluate(%d,%d,p,state,request,result,status)' % (r['provider'],r['context']),
            'if(result%has_soft.or.result%has_helicities.or.result%has_extra.or.result%has_single_helicity)stop 42',
            'alternate=state', 'state%complex_values=(0d0,0d0)',
            'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
            'if(status.ne.0)stop 32', 'call close_real(other%born,0d0)',
            'calls_before=born_test_helas_calls(%d)' % r['provider'],
            'state=alternate',
            'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
            'if(status.ne.0)stop 33', 'call close_real(result%born,other%born)',
            'if(born_test_helas_calls(%d).le.calls_before)stop 34' % r['provider'],
            'calls_first=born_test_helas_calls(%d)-calls_before' % r['provider'],
            # Transverse components alone must invalidate SAVEAMP; keep model,
            # E and pz identical, then compare with the standalone calculation.
            'pother=p', 'pother(1,:)=p(2,:)', 'pother(2,:)=p(1,:)',
            'calls_before=born_test_helas_calls(%d)' % r['provider'],
            'call born_evaluate(%d,%d,pother,state,request,other,status)' % (r['provider'],r['context']),
            'if(status.ne.0)stop 35',
            'calls_pruned=born_test_helas_calls(%d)-calls_before' % r['provider'],
            'if(calls_pruned.le.0.or.calls_pruned.gt.calls_first)stop 36',
            'nfksprocess=1', 'calculatedBorn=.false.', 'call sborn(pother,ans)',
            'call close_real(ans,other%born)',
            'call born_evaluate(%d,%d,p,state,request,result,status)' % (r['provider'],r['context'])]
        for other in contexts:
            if other['provider'] == r['provider']:
                continue
            lines += [
                'alternate%real_values=state%real_values*1.19d0',
                'alternate%complex_values=state%complex_values*(0.4d0,0.3d0)',
                'call born_evaluate(%d,%d,pother,alternate,request,other,status)' % (other['provider'],other['context']),
                'if(status.ne.0)stop 37',
                'calls_before=born_test_helas_calls(%d)' % r['provider'],
                'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
                'if(status.ne.0)stop 38', 'call close_real(result%born,other%born)',
                'if(born_test_helas_calls(%d).ne.calls_before)stop 39' % r['provider']]
        if signature:
            # A reference-model entry outside every evaluator's read set must
            # not enter the compact state or invalidate a cached amplitude.
            for kind in ('real', 'complex'):
                used = set(signature[kind]) | {index for k,index,_ in signature['couplings'] if k == kind}
                if kind == 'real':
                    used.add(signature['g'])
                size = max([int(index.split(':')[-1]) for _,k,index in members if k == kind] or [0])
                unused = sorted(set(range(1,size+1))-used)
                if not unused:
                    continue
                lines += ['alternate=state',
                    'alternate%%%s_values(%d)=alternate%%%s_values(%d)+0.125d0' %
                    (kind,unused[0],kind,unused[0]),
                    'if(.not.born_model_state_equal(state,alternate))stop 53',
                    'if(.not.born_amplitude_state_equal(state,alternate))stop 54',
                    'calls_before=born_test_helas_calls(%d)' % r['provider'],
                    'call born_evaluate(%d,%d,p,alternate,request,other,status)' % (r['provider'],r['context']),
                    'if(status.ne.0)stop 55', 'call close_real(result%born,other%born)',
                    'if(born_test_helas_calls(%d).ne.calls_before)stop 56' % r['provider'],
                    'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
                    'if(status.ne.0)stop 57', 'call close_real(result%born,other%born)',
                    'if(born_test_helas_calls(%d).ne.calls_before)stop 58' % r['provider']]
            lines += ['alternate=state',
                'state%%real_values(%d)=2d0*state%%real_values(%d)' % (signature['g'],signature['g'])]
            for kind, index, degree in signature['couplings']:
                lines += ['state%%%s_values(%d)=state%%%s_values(%d)*2d0**(%d)' %
                          (kind,index,kind,index,degree)]
            lines += ['if(.not.born_helicity_state_equal(state,alternate))stop 43',
                'if(.not.born_helicity_state_equal(state,alternate,.true.))stop 43',
                'if(.not.born_helicity_state_equal(state,state,.false.))stop 43',
                'if(born_model_state_equal(state,alternate))stop 44',
                'calls_before=born_test_helas_calls(%d)' % r['provider'],
                'call born_evaluate(%d,%d,pother,state,request,other,status)' % (r['provider'],r['context']),
                'if(status.ne.0)stop 45',
                'if(born_test_helas_calls(%d).ne.calls_before)stop 46' % r['provider']]
            lines += ['%s=state%%%s_values(%s)' % v for v in members]
            lines += ['calculatedBorn=.false.', 'nfksprocess=1', 'call sborn(pother,ans)',
                'call close_real(ans,other%born)',
                'do i=1,amp_split_size', 'call close_real(amp_split(i),other%amplitudes(i))',
                'do j=1,nsplitorders', 'do k=1,2',
                'call close_complex(amp_split_cnt(i,k,j),other%split_counterterms(i,k,j))',
                'enddo', 'enddo', 'enddo',
                'do j=1,nsplitorders', 'do k=1,2',
                'call close_complex(ans_cnt(k,j),other%counterterms(k,j))', 'enddo', 'enddo',
                'do i=1,%d' % ng, 'call close_real(amp2(i),other%diagrams(i))', 'enddo',
                'do i=1,%d' % nc, 'call close_real(jamp2(i),other%flows(i))',
                'do j=1,%d' % r['namps'],
                'call close_real(jamp2b(i,j),other%flow_orders(i,j))', 'enddo', 'enddo',
                # Return to the original coupling and momentum, as happens
                # between folds. Its saved point must still be available.
                'state=alternate',
                'calls_before=born_test_helas_calls(%d)' % r['provider'],
                'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
                'if(status.ne.0)stop 49', 'call close_real(result%born,other%born)',
                'if(born_test_helas_calls(%d).ne.calls_before)stop 50' % r['provider']]
        lines += ["write(*,*)'PASS Born provider; HELAS discovery/pruned',calls_first,calls_pruned",'end',
            'subroutine close_complex(a,b)','implicit none','complex(8) a,b',
            'call close_real(real(a,8),real(b,8))','call close_real(aimag(a),aimag(b))','end',
            'subroutine close_real(a,b)','use, intrinsic :: ieee_arithmetic','implicit none','real(8) a,b',
            'if(.not.ieee_is_finite(a).or..not.ieee_is_finite(b))stop 21',
            'if(abs(a-b).gt.1d-12*max(1d-50,abs(a),abs(b)))then',
            "write(*,*)'Mismatch',a,b",'error stop 22','endif','end',
            'double precision function ran2()', 'stop 23', 'end']
        return (model_state_checks.adapt_wide_driver(lines, members, sizes, layout)
                if layout else lines)

    def test_dy(self):
        self.compare(self.outputs['dy'])

    def test_standard_exports_retain_exact_tree_cache(self):
        for name in (name for name in ('dy','ttbar','wjet') if name in self.outputs):
            with self.subTest(process=name):
                signature=json.loads((self.outputs[name]/'Source/BornSupport/helicity_signature.json').read_text())
                self.assertIn(signature['mode'], ('homogeneous_G', 'exact_compact_state'))
                if signature['mode']=='homogeneous_G':
                    self.assertIsNotNone(signature['signature'])
                else:
                    # FLV_COUPLING graphs keep exact numeric target inputs
                    # until their G homogeneity has a complete proof.
                    layout=json.loads((self.outputs[name]/'Source/BornSupport/model_state_layout.json').read_text())
                    registry=json.loads((self.outputs[name]/'Source/BornSupport/registry.json').read_text())
                    inputs={coupling for record in registry['contexts']
                            for matrix in record['helicity_matrices']
                            for coupling in matrix.get('couplings', {})}
                    members={name for name,_,_ in layout['fixed_members']+layout['dynamic_members']}
                    self.assertTrue(inputs & members)

    def test_ttbar(self):
        self.compare(self.outputs['ttbar'])

    def test_wjet(self):
        self.compare(self.outputs['wjet'])

    def test_extra_born(self):
        self.compare(self.outputs['extra'])

    def test_mixed_orders(self):
        self.compare(self.outputs['mixed'])

    def test_gluon_colour_density(self):
        self.compare(self.outputs['gluons'])

    def test_qed_charge_correlations(self):
        self.compare(self.outputs['qed'])

    def test_loonly(self):
        self.compare(self.outputs['loonly'])

    def test_real_history_cache(self):
        from tests.unit_tests.iolibs.native_cache_checks import check_real_cache
        for name in ('dy','ttbar','wjet'):
            with self.subTest(process=name):
                check_real_cache(self.outputs[name],self.run_command)

    def test_native_context_switching(self):
        """Bounds-checked native aliases reproduce explicit provider requests."""
        for name in ('dy','ttbar','wjet','loonly'):
            output = self.outputs[name]
            records = json.loads((output/'Source/BornSupport/registry.json').read_text())['contexts']
            model = support.expand(output/'Source/MODEL/coupl.inc')
            model += support.expand(output/'Source/MODEL/input.inc')
            members, sizes = support.common_members(model)
            layout = model_state_checks.load_layout(output)
            for record in records:
                directory = output/'SubProcesses'/record['directory']
                lines = ['program check_native','use mc_native_context',
                    'use mc_counterterms, only: set_QCD_flows','implicit none',
                    "include 'nexternal.inc'", "include 'genps.inc'", "include 'orders.inc'",
                    "include 'born_nhel.inc'", "include 'born_conf.inc'",
                    'integer idup(nexternal-1,maxproc),mothup(2,nexternal-1,maxproc)',
                    'integer icolup(2,nexternal-1,max_bcol)',"include 'born_leshouche.inc'",
                    'integer nfksprocess,status,i,j,k,iteration,c',
                    'common/c_nfksprocess/nfksprocess',
                    'real(8) p(0:3,nexternal-1),ans,soft(amp_split_size),mc_born_flow_weight',
                    'real(8) charges(nexternal-1)', 'common/c_charges_born/charges',
                    'common/to_amp_split_soft/soft',
                    'logical need_color_links,need_charge_links',
                    'common/c_need_links/need_color_links,need_charge_links',
                    'logical split_type(nsplitorders),is_leading_cflow(max_bcol)',
                    'integer num_leading_cflows',
                    'common/c_split_type/split_type',
                    'common/c_leading_cflows/is_leading_cflow,num_leading_cflows',
                    'type(BornModelState) state','type(BornRequest) request',
                    'type(BornResult) reference']+model+[
                    'allocate(state%%real_values(%d),state%%complex_values(%d))' %
                        (sizes['real'],sizes['complex']),
                    'state%real_values=1.1d0','state%complex_values=(0.7d0,0.2d0)',
                    'charges=0.5d0','request%charges=charges',
                    'p=0d0','p(:,1)=[500d0,0d0,0d0,500d0]',
                    'p(:,2)=[500d0,0d0,0d0,-500d0]',
                    'p(:,3)=[500d0,300d0,0d0,400d0]',
                    'p(:,4)=[500d0,-300d0,0d0,-400d0]',
                    'do iteration=1,3','state%complex_values=state%complex_values*(0.9d0,0.1d0)',
                    'p(1:2,:)=-p(1:2,:)']
                lines += ['%s=state%%%s_values(%s)' % v for v in members]
                lines += ['call mc_born_model_changed()']
                lines += ['do nfksprocess=1,size(native_sector_ids)',
                    'call activate_native_context(nfksprocess)',
                    'c=active_context', 'request%sector=active_sector',
                    'request%colour=.false.', 'need_color_links=.true.',
                    'need_charge_links=.false.',
                    'call sborn_native(p,ans)',
                    'call born_evaluate(context_providers(c),c,p,state,request,reference,status)',
                    'if(status.ne.0)stop 11','call close_real(ans,reference%born)',
                    'if(mapconfig(0).ne.native_metadata%configurations(0))stop 12',
                    'if(any(idup(:,1:iproc_born).ne.native_metadata%born_ids))stop 13',
                    'split_type=.false.', 'is_leading_cflow=.false.',
                    'call set_QCD_flows',
                    'if(num_leading_cflows.le.0)stop 16',
                    'if(.not.any(is_leading_cflow))stop 17',
                    'do i=1,native_metadata%ncolor',
                    'call close_real(mc_born_flow_weight(i),reference%flows(i))','enddo',
                    'do i=1,native_metadata%namplitudes',
                    'j=native_amplitude_map(i,c)','if(j.eq.0)cycle',
                    'call close_real(amp_split(j),reference%amplitudes(i))','enddo',
                    # A correlated call must not overwrite the caller's accumulated
                    # soft amplitude. This is distinct from the provider's cache.
                    'if(%s)then' % ('.true.' if record['sectors'] else '.false.'),
                    'amp_split=123d0','call sborn_sf_native(p,1,2,ans)',
                    'if(any(amp_split.ne.123d0))stop 14',
                    'request%colour=.true.','request%m=1','request%n=2',
                    'call born_evaluate(context_providers(c),c,p,state,request,reference,status)',
                    'if(status.ne.0)stop 15','call close_real(ans,reference%correlation)',
                    'do i=1,native_metadata%namplitudes',
                    'j=native_amplitude_map(i,c)','if(j.eq.0)cycle',
                    'call close_real(soft(j),reference%soft(i))','enddo','endif',
                    'enddo','call activate_native_context(1)','enddo',
                    "write(*,*)'PASS native contexts'",'end']
                checks = self.driver(record,members,sizes,model,records)
                lines += checks[checks.index('subroutine close_complex(a,b)'):]
                lines = model_state_checks.adapt_wide_driver(lines, members, sizes, layout)
                support.write_fortran(directory/'check_native.f',lines)
                support.write_fortran(directory/'model_revision.f',
                                      model_state_checks.revision_helpers(output))
                # Keep pure order helpers only, as in the standalone comparison.
                support.write_fortran(directory/'native_helpers.f',sum((support.routine(
                    (directory/'splitorders_stuff.f').read_text(),n) for n in
                    ('orders_to_amp_split_pos','amp_split_pos_to_orders')),[]))
                counter = Path(support.__file__).resolve().parents[2]/'Template/NLO/SubProcesses/montecarlocounter.f'
                (directory/'native_flows.f').write_text(mc_counterterm_test_module(
                    ('set_QCD_flows', 'check_QCD_flows'), counter.parent,
                    include_kinematics=False))
                # Flow setup imports process state; compile its QED support
                # dependency too, as in the production subprocess build.
                # The runtime's channel-weight helper imports active point data.
                from tests.unit_tests.fks.mint_fixture import mint_runtime_modules
                paths = [*mint_runtime_modules(directory),
                         '-x','f77',output/'Source/MODEL/coupl.f90','-x','none',
                         'qed_shower_support.f90','process_module.f90',
                         'fks_phase_space_data.f','mc_native_context.f90',
                         'mc_native_runtime.f','mc_native_props.f',
                         'born_support.f','born.f','sborn_sf.f','extra_cnt_wrapper.f',
                         'native_helpers.f','native_flows.f','model_revision.f','check_native.f']
                self.run_command(['gfortran','-O0','-g','-fcheck=all',
                    '-ffixed-line-length-none','-ffree-line-length-none','-ffunction-sections',
                    '-I'+str(output/'lib'),'-I'+str(output/'Source/BornSupport'),
                    str(output/'Source/timing.f'),*paths,
                    '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections',
                    '-L'+str(output/'lib'),'-lmc_born_support','-Wl,-rpath,'+self.origin+'/../../lib',
                    '-o','check_native'],directory)
                self.assertIn('PASS native contexts',self.run_command(['./check_native'],directory))

    def test_shared_provider_retains_both_contexts(self):
        output = self.outputs['shared']
        support_dir = output/'Source/BornSupport'
        records = json.loads((support_dir/'registry.json').read_text())['contexts']
        self.assertEqual(len(records),2)
        self.assertEqual({r['provider'] for r in records},{1})
        self.assertEqual({r['context'] for r in records},{1,2})
        self.assertEqual(len(list(support_dir.glob('p*/matrix.f'))),1)
        self.assertEqual(len(list(support_dir.glob('c*/metadata.f'))),2)
        for r in records:
            wrapper = output/'SubProcesses'/r['directory']/'born.f'
            scalar_body = '\n'.join(support.routine(wrapper.read_text(),'sborn')).upper()
            self.assertIn('CALL MC_BORN_LOCAL(', scalar_body)
            self.assertNotIn('CALL SBORN_SPLITORDERS(', scalar_body)
        self.compare(output)

    def test_parallel_registry(self):
        from madgraph.interface import master_interface
        interface = master_interface.MasterCmd()
        output = self.work/'dy_parallel'
        for command in ('set automatic_html_opening False --no_save',
                        'set low_mem_multicore_nlo_generation True --no_save',
                        'set nb_core 2 --no_save', 'import model sm',
                        'define p = g u u~ d d~',
                        'generate p p > e+ e- QED^2=4 QCD^2=0 [real=QCD]',
                        'output %s -f -nojpeg' % output):
            interface.exec_cmd(command,errorhandling=False,printcmd=False,
                               precmd=True,postcmd=True)
        registry = Path('Source/BornSupport/registry.json')
        self.assertEqual(json.loads((output/registry).read_text()),
                         json.loads((self.outputs['dy']/registry).read_text()))

    def test_incremental_and_hidden_symbols(self):
        output = self.outputs['ttbar']
        directory = output/'Source/BornSupport'
        self.run_command(['make','-q'],directory)
        os.utime(directory/'p1/matrix.f',None)
        commands = self.run_command(['make','-n'],directory)
        self.assertIn('-c p1/matrix.f',commands)
        self.assertNotIn('-c p2/matrix.f',commands)
        self.run_command(['make','FFLAGS=-O0 -g -fcheck=all -fbacktrace'],directory)
        dependency = next((output/'Source/DHELAS').glob('FFV*.f'))
        os.utime(dependency,None)
        commands = self.run_command(['make','-n'],directory)
        self.assertIn('refresh_dependency.py',commands)
        self.assertNotIn('-c p1/matrix.f',commands)
        self.run_command(['make','FFLAGS=-O0 -g -fcheck=all -fbacktrace'],directory)
        self.run_command(['make','-q'],directory)
        (output/'lib/mc_born_support.mod').unlink()
        commands = self.run_command(['make','-n'],directory)
        self.assertIn('cp mc_born_support.mod',commands)
        self.assertNotIn(' -c ',commands)
        self.run_command(['make'],directory)
        self.run_command(['make','-q'],directory)
        if os.name == 'posix' and shutil.which('nm') and (output/'lib/libmc_born_support.so').exists():
            symbols = self.run_command(['nm','-D','--defined-only',str(output/'lib/libmc_born_support.so')],directory)
            for line in symbols.splitlines():
                self.assertTrue(line.split()[-1].startswith(('__mc_born_support_MOD_','__mc_born_types_MOD_')),line)

    def test_z_relocation(self):
        original = self.outputs['ttbar']
        moved = self.work/'relocated'
        original.rename(moved)
        for executable in moved.glob('reference/P*/check'):
            self.assertIn('PASS Born provider',self.run_command(['./check'],executable.parent))
