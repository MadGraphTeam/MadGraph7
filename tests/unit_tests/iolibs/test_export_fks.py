################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Unit test library for the export_FKS format routines"""

from __future__ import absolute_import
import copy
import filecmp
import fractions
import os
import re
import sys
import tempfile
import glob
import shutil
import subprocess
from types import SimpleNamespace
from tests import test_manager

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.various.misc as misc
import madgraph.iolibs.files as files
import tests.IOTests as IOTests
import madgraph.interface.master_interface as MGCmd

import madgraph.fks.fks_common as fks_common
import madgraph.fks.fks_helas_objects as fks_helas_objects
import madgraph.fks.fks_helas_objects as fks_helas
import madgraph.core.base_objects as base_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.export_fks as export_fks
import madgraph.iolibs.export_v4 as export_v4
import madgraph.iolibs.file_writers as file_writers
import madgraph.loop.loop_exporters as loop_exporters
import madgraph.iolibs.file_writers as writers
from madgraph import MadGraph5Error

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                'input_files')

class TestBornDirCollision(unittest.TestCase):
    """P<...> directory names must never be shared by two different matrix
    elements.

    shell_string() concatenates particle names and polarization labels with
    no separator, and every FKS born process carries id 0, so a clash is
    possible for some models.  os.mkdir always raises EEXIST, so the base
    died loudly but opaquely; the guard names both processes instead.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp('', 'TMPBornDir', None)
        self.cwd = os.getcwd()
        os.chdir(self.tmpdir)
        self.model = base_objects.Model()
        self.model.set('particles', base_objects.ParticleList([
            base_objects.Particle({'name': 'c', 'antiname': 'c~',
                                   'pdg_code': 3})]))

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmpdir)

    def make_process(self, polarization):
        legs = base_objects.LegList(
            [base_objects.Leg({'id': 3, 'number': i + 1, 'state': i > 1})
             for i in range(4)])
        legs[2].set('polarization', polarization)
        return base_objects.Process({'legs': legs, 'model': self.model,
                                     'id': 0})

    def test_born_dir_collision_is_explicit(self):
        """A second matrix element asking for a taken directory must get an
        error naming both processes, not a bare FileExistsError."""

        exporter = export_fks.ProcessExporterFortranFKS.__new__(
                                    export_fks.ProcessExporterFortranFKS)

        first = self.make_process([1])
        second = self.make_process([-1])
        # force the clash: whatever shell_string() does, both want 'P0_clash'
        exporter.mkdir_born_dir('P0_clash', first)
        self.assertTrue(os.path.isdir('P0_clash'))

        try:
            exporter.mkdir_born_dir('P0_clash', second)
        except MadGraph5Error as error:
            message = str(error)
        else:
            self.fail('no error raised on a P directory name collision')

        self.assertIn('P0_clash', message)
        self.assertIn(first.nice_string(prefix=False).strip(), message)
        self.assertIn(second.nice_string(prefix=False).strip(), message)

    def test_born_dir_no_collision(self):
        """Distinct names are created without complaint, and the guard does
        not interfere with an ordinary (unpolarized) name."""

        exporter = export_fks.ProcessExporterFortranFKS.__new__(
                                    export_fks.ProcessExporterFortranFKS)
        plain = self.make_process([])
        self.assertEqual('0_cc_cc', plain.shell_string())
        exporter.mkdir_born_dir('P%s' % plain.shell_string(), plain)

        # 'c{S,S}' and 'c{A}' are different restrictions and used to render
        # the same string; they must now get one directory each
        left = self.make_process([9, 9])
        right = self.make_process([99])
        self.assertNotEqual(left.shell_string(), right.shell_string())
        exporter.mkdir_born_dir('P%s' % left.shell_string(), left)
        exporter.mkdir_born_dir('P%s' % right.shell_string(), right)
        self.assertEqual(sorted(os.listdir('.')),
                         ['P0_cc_c99c', 'P0_cc_c9c', 'P0_cc_cc'])


class TestGroupedFKSMetadata(unittest.TestCase):
    """Physical FKS classes exported through configuration-indexed tables."""

    def test_native_mc_keeps_ordered_point_dispatch_and_openmp_build_flags(self):
        """Native MC folds share typed state; retain ordered full points.

        Fortran amplitude helpers still support explicit OpenMP builds.
        Porting native MC point state to lanes is a separate change.
        """

        template_root = os.path.join(
            root_path, os.path.pardir, os.path.pardir, 'Template', 'NLO')
        with open(os.path.join(
                template_root, 'SubProcesses', 'driver_mintMC.f')) as stream:
            driver = stream.read().upper()
        self.assertIn('CALL MINT(SIGINTF)', driver)
        self.assertIn('CALL GEN(SIGINTF,', driver)
        self.assertNotIn('CALL MINT_VEC(', driver)
        self.assertNotIn('CALL GEN_VEC(', driver)

        with open(os.path.join(
                template_root, 'Source', 'make_opts.inc')) as stream:
            make_opts = stream.read()
        self.assertIn('ifeq ($(openmp),true)', make_opts)
        self.assertIn('FFLAGS += -fopenmp', make_opts)
        self.assertIn('LDFLAGS += -fopenmp', make_opts)
        self.assertIn('OMP_RECURSIVE_FFLAGS', make_opts)
        self.assertIn('filter-out -fno-automatic', make_opts)

        with open(os.path.join(
                template_root, 'SubProcesses', 'makefile_fks_dir')) as stream:
            makefile = stream.read()
        self.assertIn('OPENMP_RECURSIVE_OBJS', makefile)
        self.assertIn('driver_vec.o real_me_chooser.o born.o', makefile)
        self.assertIn('$(OMP_RECURSIVE_FFLAGS)', makefile)

        aloha_root = os.path.join(
            root_path, os.path.pardir, os.path.pardir, 'aloha',
            'template_files')
        for filename in ('Makefile_F', 'Makefile_F_dual'):
            with open(os.path.join(aloha_root, filename)) as stream:
                aloha_makefile = stream.read()
            self.assertIn('ifeq ($(openmp),true)', aloha_makefile)
            self.assertIn('$(OMP_RECURSIVE_FFLAGS)', aloha_makefile)

        build_script = os.path.join(
            template_root, 'Utilities', 'build_openmp_benchmark.sh')
        with open(build_script) as stream:
            script = stream.read()
        self.assertIn('Source/DHELAS', script)
        self.assertIn('openmp=true', script)
        self.assertIn('driver_mintMC.o', script)

    def test_standalone_exporters_follow_loop_optimization(self):
        self.assertTrue(issubclass(
            export_fks.ProcessExporterFortranFKS_SA,
            export_fks.ProcessOptimizedExporterFortranFKS))
        self.assertTrue(issubclass(
            export_fks.ProcessExporterFortranFKS_SA_Default,
            export_fks.ProcessExporterFortranFKS))
        self.assertFalse(issubclass(
            export_fks.ProcessExporterFortranFKS_SA_Default,
            export_fks.ProcessOptimizedExporterFortranFKS))

    def test_openmp_amplitude_locals_are_not_saved_by_data(self):
        """Grouped runtime flavor selection must remain lane-local."""

        template_dir = os.path.join(
            root_path, os.path.pardir, os.path.pardir,
            'madgraph', 'iolibs', 'template_files')
        for filename in ('realmatrix_splitorders_fks.inc',
                         'bornmatrix_splitorders_fks.inc',
                         'born_cnt_splitorders_fks.inc'):
            with open(os.path.join(template_dir, filename)) as stream:
                content = stream.read().upper()
            self.assertNotIn('DATA FLAVOR', content)
            self.assertIn('FLAVOR(:) = 1', content)

    def test_sudakov_virtual_calls_use_fks_flavor(self):
        """Sudakov diagnostics must not fall back to virtual row one."""

        template_dir = os.path.join(
            root_path, os.path.pardir, os.path.pardir,
            'Template', 'NLO', 'SubProcesses')
        for filename in ('check_sudakov.f', 'check_sudakov_angle2.f'):
            with open(os.path.join(template_dir, filename)) as stream:
                content = stream.read().upper()
            self.assertIn("INCLUDE 'FKS_INFO.INC'", content)
            self.assertIn(
                'VIRTUAL_FLAVOR=VIRTUAL_FLAVOR_INDEX_D(NFKSPROCESS)',
                content)
            self.assertNotRegex(content,
                                r'CALL\s+SLOOPMATRIX(?:HEL)?_THRES\(')
            self.assertIn('CALL SLOOPMATRIX_THRES_FLAVOR(', content)
            self.assertIn('CALL SLOOPMATRIXHEL_THRES_FLAVOR(', content)

    def test_grouped_virtual_rows_sample_their_own_helicities(self):
        """Every virtual row of a P directory does its own MC over helicities.

        MadLoop keeps one helicity filter per evaluated row when there are
        several (fks_info.inc: HELFILTER_PER_VIRTUAL_FLAVOR). BinothLHA then
        reads the filter of the row it evaluates, and keeps the helicities
        and the MC-or-sum choice of each row apart. Its explicit sums pass
        HEL=-1, which resets the helicity that the MC over helicities of
        another row leaves in MadLoop. The drivers only fall back to the
        explicit sum for several rows without per-row filters.
        """

        template_dir = os.path.join(
            root_path, os.path.pardir, os.path.pardir,
            'Template', 'NLO', 'SubProcesses')
        for filename in ('driver_mintFO.f', 'driver_mintMC.f'):
            with open(os.path.join(template_dir, filename)) as stream:
                content = ''.join(stream.read().upper().replace(
                    '\n     $', '').split())
            self.assertIn('I.EQ.0.OR.(N_VIRTUAL_FLAVOR_CONFIGS.GT.1.AND..NOT.'
                          'HELFILTER_PER_VIRTUAL_FLAVOR)', content)

        with open(os.path.join(template_dir, 'BinothLHA.f')) as stream:
            content = stream.read().upper()
        # BinothLHA_vec is not called: check the live subroutine only.
        code = '\n'.join(
            line for line in
            content[:content.index('SUBROUTINE BINOTHLHA_VEC(')].split('\n')
            if not line.startswith('C'))
        live = ''.join(code.replace('\n     $', '').replace('\n     &', '')
                       .split())
        self.assertIn("IF(HELFILTER_PER_VIRTUAL_FLAVOR)THENWRITE(HELFILTER_FILE,"
                      "'(A,I0,A)')'../MADLOOP5_RESOURCES/HELFILTER_',"
                      "VIRTUAL_FLAVOR,'.DAT'", live)
        self.assertIn('INTEGERGOODHEL(MAX_BHEL,MAX_VIRTUAL_FLAVOR_INDEX)',
                      live)
        self.assertIn('CALLPICKHELICITYMC(P,GOODHEL(:,VIRTUAL_FLAVOR),'
                      'HEL(:,VIRTUAL_FLAVOR),', live)
        self.assertFalse('MC_HEL=0' in live)
        self.assertFalse('CALLSLOOPMATRIX_THRES_FLAVOR(' in live)
        self.assertEqual(live.count(
            'CALLSLOOPMATRIXHEL_THRES_FLAVOR(P,-1,VIRTUAL_FLAVOR,'), 2)

        # check_poles initialises every row, so that each filter is written
        # before the integration jobs need it.
        with open(os.path.join(template_dir, 'check_poles.f')) as stream:
            content = ''.join(stream.read().upper().replace(
                '\n     $', '').split())
        self.assertIn('IF(NPOINTSCHECKED.EQ.0.OR.HELFILTER_PER_VIRTUAL_FLAVOR)'
                      'THEN', content)
        self.assertIn('IF(HELFILTER_PER_VIRTUAL_FLAVOR.AND.NPOINTSCHECKED.LT.'
                      'NBORN_FLAVOR_CONFIGS)GOTO200', content)

    def test_grouped_virtual_rows_keep_their_own_madloop_filters(self):
        """MadLoop keeps filters for the virtual rows that a P directory uses.

        The virtuals of e+ e- > t t~ and mu+ mu- > t t~ group e and mu, but
        each P directory evaluates a single row: its filters describe that
        row, as in an ungrouped output. The P directories of p p > e+ ve
        evaluate two rows each: every row then gets its own filter slot
        (FILTER_FLAV_IDX=-1), and only those rows. Without the rows of its
        caller (standalone output) a grouped virtual keeps its filters off.
        """

        template_dir = os.path.join(
            root_path, os.path.pardir, os.path.pardir,
            'madgraph', 'iolibs', 'template_files')
        for loop_dir in ('loop', 'loop_optimized'):
            with open(os.path.join(template_dir, loop_dir,
                                   'loop_matrix_standalone.inc')) as stream:
                content = ' '.join(stream.read().upper().split())
            self.assertIn('IF (FILTER_FLAV_IDX.EQ.0) THEN C A SINGLE MADLOOP '
                          'FILTER FILE', content)
            self.assertIn('IF (FILTER_FLAV_IDX.GT.0.AND.FLAV_IDX.NE.'
                          'FILTER_FLAV_IDX) THEN', content)
            self.assertNotIn('NVIRTUAL_FLAVORS.GT.1', content)

        get_rows = export_fks.ProcessExporterFortranFKS.\
            get_virtual_evaluated_flavor_pdgs
        exporter_class = loop_exporters.LoopProcessExporterFortranSA
        expected = {'e+ e- > t t~ [QCD]': {'P0_epem_ttx': 1},
                    'mu+ mu- > t t~ [QCD]': {'P0_mupmum_ttx': 2},
                    'p p > e+ ve [QCD]': {'P0_QQx_veep': -1,
                                          'P0_QxQ_veep': -1}}
        for process, filters in expected.items():
            interface = MGCmd.MasterCmd()
            interface._experimental_nlo_flavor_grouping = True
            interface.no_notification()
            interface.exec_cmd('import model loop_sm')
            interface.exec_cmd('generate %s' % process)
            helas = fks_helas_objects.FKSHelasMultiProcess(
                interface._fks_multi_proc)
            found = {}
            for matrix_element in helas.get_matrix_elements():
                virtual = matrix_element.virt_matrix_element
                if not virtual._flavor_is_populated():
                    virtual.populate_flavor_validity()
                n_flavors = len(virtual.get('allowed_flavors'))
                self.assertGreater(n_flavors, 1)
                # Without the rows of its caller the filters stay off.
                self.assertIsNone(exporter_class.get_virtual_filter_rows(
                    virtual))
                self.assertEqual(exporter_class.get_virtual_filter_flavor(
                    virtual, n_flavors), 0)
                # The rows are set before the virtual is exported, as in
                # generate_directories_fks.
                virtual.evaluated_flavor_pdgs = get_rows(matrix_element)
                exporter = exporter_class.__new__(exporter_class)
                replace_dict = {'proc_prefix': ''}
                exporter.set_virtual_flavor_replace_dict_entries(
                    replace_dict, virtual)
                filter_flavor = replace_dict['filter_flav_idx']
                self.assertEqual(virtual.virtual_filter_flavor,
                                 filter_flavor)
                found['P%s' % matrix_element.born_me.get(
                    'processes')[0].shell_string()] = filter_flavor
                # The rows the P directory evaluates at run time.
                virtual_rows = sorted(set(
                    info['flavor_class']['virtual_flavor_index']
                    for info in matrix_element.get_fks_info_list()))
                slots = [int(slot) for slot in
                         replace_dict['filter_slots'].split(',')]
                self.assertEqual(len(slots), n_flavors)
                if filter_flavor > 0:
                    self.assertEqual(virtual_rows, [filter_flavor])
                    self.assertEqual(replace_dict['nfilter_slots'], 1)
                    self.assertEqual(set(slots), set([0]))
                else:
                    self.assertGreater(len(virtual_rows), 1)
                    self.assertEqual(exporter_class.get_virtual_filter_rows(
                        virtual), virtual_rows)
                    self.assertEqual(replace_dict['nfilter_slots'],
                                     len(virtual_rows))
                    self.assertEqual(
                        [row for row, slot in enumerate(slots, 1) if slot],
                        virtual_rows)
                    self.assertEqual([slot for slot in slots if slot],
                                     list(range(1, len(virtual_rows) + 1)))
                # The per-row filter code is only written for -1.
                exporter.cmd_options = {}
                exporter.compute_color_flows = False
                exporter.has_loop_induced = False
                self.assertEqual(
                    exporter_class.get_context(exporter, virtual)[
                        'FilterPerFlavor'], filter_flavor == -1)
            self.assertEqual(found, filters, process)

    def test_per_row_filter_code_is_only_written_for_several_rows(self):
        """MadLoop only gets the per-row filter code for FILTER_FLAV_IDX=-1.

        The loop_matrix.f of standalone outputs and of P directories with
        one virtual row is then unchanged.
        """

        template_dir = os.path.join(
            root_path, os.path.pardir, os.path.pardir,
            'madgraph', 'iolibs', 'template_files')
        context = {'LoopInduced': False, 'ComputeColorFlows': False,
                   'AmplitudeReduction': False, 'TIRCaching': False,
                   'MadEventOutput': False, 'UseDensity': False,
                   'collier_available': True, 'iregi_available': False,
                   'ninja_available': True, 'ninjars_available': False,
                   'ninja_supports_quad_prec': False, 'ninjars_supports_quad_prec': False,
                   'tirust_available': False, 'tirust_supports_quad_prec': False}
        per_row_names = ('NFILTER_SLOTS', 'FILTER_SLOT', 'LOADED_FILTER_FLAV',
                         "'HelFilter_'", "'LoopFilter_'")
        for loop_dir in ('loop', 'loop_optimized'):
            with open(os.path.join(template_dir, loop_dir,
                                   'loop_matrix_standalone.inc')) as stream:
                template = stream.read()
            writer = file_writers.FortranWriter(os.devnull)
            for per_row in (False, True):
                lines = writer.preprocess_template(
                    template, dict(context, FilterPerFlavor=per_row))
                code = '\n'.join(lines)
                for name in per_row_names:
                    self.assertEqual(name in code, per_row, (loop_dir, name))
                if per_row:
                    # The state of a row is swapped in before the filters
                    # are read, and the filter files are named after it.
                    self.assertLess(code.index('LOADED_FILTER_FLAV=FLAV_IDX'),
                                    code.index('IF(NTRY.EQ.0) THEN'))
                    for state in ('NTRY', 'NATTEMPTS', 'FOUNDHELFILTER',
                                  'FOUNDLOOPFILTER', 'CHECKPHASE',
                                  'HELDOUBLECHECKED', 'DONEHELDOUBLECHECK'):
                        self.assertIn('SLOT_%s(ISLOT)=%s' % (state, state),
                                      code)
                        self.assertIn('%s=SLOT_%s(ISLOT)' % (state, state),
                                      code)
            writer.close()

    def test_fixed_order_stratifies_physical_flavors_by_topology(self):
        """Grouped FO sampling sums flavours inside one sampled topology."""

        template_dir = os.path.join(
            root_path, os.path.pardir, os.path.pardir,
            'Template', 'NLO', 'SubProcesses')
        with open(os.path.join(template_dir, 'driver_mintFO.f')) as stream:
            content = ' '.join(
                stream.read().upper().replace('$', '').split())

        self.assertIn(
            'CALL SETUP_PHYSICAL_FKS_MAP(PHYSICAL_FKS_GROUP_COUNT, '
            'PHYSICAL_FKS_GROUP_SIZE,PHYSICAL_FKS_GROUP_OFFSET, '
            'PHYSICAL_FKS_MEMBERS)', content)
        self.assertIn(
            'IGROUP=TOPOLOGY_GROUP(FKS_TOPOLOGY_D(IFKS))', content)
        self.assertIn('DO IFLAV_CONFIG=1,NBORN_FLAVOR_CONFIGS', content)
        self.assertIn(
            'NFKS_BORN=BORN_FKS_CONFIG_D(IFLAV_CONFIG)', content)
        self.assertIn('IF (SUM) THEN NFKS_MIN=1 NFKS_MAX=', content)
        self.assertIn('DO IGROUP=NFKS_MIN,NFKS_MAX', content)
        self.assertIn(
            'DO J=1,PHYSICAL_FKS_GROUP_SIZE(IGROUP,NFKS_SECTOR)',
            content)
        self.assertIn(
            'IFKS=PHYSICAL_FKS_MEMBERS(I,NFKS_SECTOR)', content)
        self.assertNotIn('PHYSICAL_FKS_MAP(0:FKS_CONFIGS,', content)
        self.assertIn(
            'IF (BORN_FLAVOR_INDEX_D(IFKS).NE.BORN_CLASS) CYCLE',
            content)
        self.assertIn('MC_INT_WGT=1D0/VOL', content)
        self.assertIn('JAC=MC_INT_WGT', content)
        self.assertIn('SIG_NO_NBODY=SIG_NO_NBODY+SIG', content)
        self.assertIn('ABS(SIG_NO_NBODY)*VOL)', content)
        self.assertNotIn('GET_MC_INTEGER_GROUP_VOLUME', content)

    def test_fixed_order_sreal_store_uses_current_interface(self):
        """FO calls must keep the stored and returned amplitudes distinct.

        ``sreal_store`` gained a ``real_amp_split`` input after its returned
        amplitude.  A missing argument is not diagnosed without an explicit
        Fortran interface and shifts every subsequent array argument.
        """

        template_path = os.path.join(
            root_path, os.path.pardir, os.path.pardir,
            'Template', 'NLO', 'SubProcesses', 'driver_mintFO.f')
        with open(template_path) as stream:
            content = ''.join(
                stream.read().upper().replace('$', '').split())

        self.assertEqual(content.count('CALLSREAL_STORE_BUFFERED('), 8)
        self.assertEqual(
            content.count('RET_AMP_SPLIT,REAL_AMP_SPLIT'), 8)
        for routine in (
                'COMPUTE_SOFT_COUNTER_TERM',
                'COMPUTE_SOFT_COLLINEAR_COUNTER_TERM',
                'COMPUTE_COLLINEAR_COUNTER_TERM',
                'COMPUTE_REAL_EMISSION'):
            calls = re.findall(
                r'CALL%s_BUFFERED\(([^)]*)\)' % routine, content)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(
                'RET_AMP_SPLIT' in call for call in calls))

    def test_grouped_poles_use_physical_born_classes(self):
        """Pole checks select links and virtuals from each physical class."""

        template_dir = os.path.join(
            root_path, os.path.pardir, os.path.pardir,
            'Template', 'NLO', 'SubProcesses')
        with open(os.path.join(template_dir, 'fks_singular.f')) as stream:
            content = ' '.join(stream.read().upper().replace('$', '').split())
        getpoles = content.split('SUBROUTINE GETPOLES', 1)[1].split(
            'SUBROUTINE SETFKSFACTOR', 1)[0]
        self.assertIn("INCLUDE 'FKS_INFO.INC'", getpoles)
        self.assertIn(
            'BORN_FLAVOR_INDEX_D(NFKSPROCESS).NE. '
            'BORN_FLAVOR_INDEX_D(NFKSPROCESS_SAVE)', getpoles)

        with open(os.path.join(template_dir, 'check_poles.f')) as stream:
            content = ' '.join(stream.read().upper().replace('$', '').split())
        self.assertIn("INCLUDE 'FKS_INFO.INC'", content)
        self.assertIn(
            'NFKSPROCESS=BORN_FKS_CONFIG_D( '
            'MOD(NPOINTSCHECKED,NBORN_FLAVOR_CONFIGS)+1)', content)
        self.assertIn('VIRTUAL_FLAVOR_INDEX_D(NFKSPROCESS)', content)

    def test_physical_denominator_factors(self):
        """Merged rows retain their own final-state symmetry factors."""

        process = base_objects.Process({
            'legs': base_objects.LegList([
                base_objects.Leg({'id': 81, 'state': False, 'number': 1}),
                base_objects.Leg({'id': 81, 'state': False, 'number': 2}),
                base_objects.Leg({'id': 24, 'state': True, 'number': 3}),
                base_objects.Leg({'id': 81, 'state': True, 'number': 4}),
                base_objects.Leg({'id': 81, 'state': True, 'number': 5}),
            ])})

        class MatrixElementStub(dict):
            def get_denominator_factor(self):
                return 72

            def get_external_flavors(self, return_pdgs=False):
                rows = [[1, 2, 24, 1, 1], [2, 2, 24, 1, 2]]
                return ([None, None], rows) if return_pdgs else [None, None]

        matrix_element = MatrixElementStub({
            'processes': [process], 'identical_particle_factor': 2})
        exporter_class = export_fks.ProcessExporterFortranFKS
        get_factors = exporter_class.get_flavor_denominator_factors
        self.assertEqual(
            get_factors(matrix_element),
            [72, 36])

    def test_write_fks_info_uses_physical_flavor_classes(self):
        interface = MGCmd.MasterCmd()
        interface._experimental_nlo_flavor_grouping = True
        interface.no_notification()
        interface.exec_cmd('import model loop_sm')
        interface.exec_cmd('generate p p > w+ w- [QCD]')
        helas = fks_helas_objects.FKSHelasMultiProcess(
            interface._fks_multi_proc)
        matrix_element = helas.get_matrix_elements()[0]

        info_list = matrix_element.get_fks_info_list()
        self.assertEqual(len(info_list), 16)
        self.assertTrue(all('flavor_class' in info for info in info_list))
        self.assertTrue(all(abs(pdg) not in
                            interface._curr_model.get('merged_particles')
                            for info in info_list for pdg in info['pdgs']))

        exporter = export_fks.ProcessExporterFortranFKS.__new__(
            export_fks.ProcessExporterFortranFKS)

        def write_fks_info():
            handle, path = tempfile.mkstemp(prefix='fks_info_',
                                            suffix='.inc')
            os.close(handle)
            try:
                writer = file_writers.FortranWriter(path)
                exporter.write_fks_info_file(
                    writer, matrix_element, None)
                writer.close()
                with open(path) as stream:
                    return stream.read()
            finally:
                if os.path.exists(path):
                    os.remove(path)

        # BinothLHA reads one helicity filter per virtual row only once
        # MadLoop was exported with one filter slot per row.
        virtual = matrix_element.virt_matrix_element
        for filter_flavor, flag in ((None, '.FALSE.'), (0, '.FALSE.'),
                                    (-1, '.TRUE.')):
            if filter_flavor is not None:
                virtual.virtual_filter_flavor = filter_flavor
            content = write_fks_info()
            self.assertIn('PARAMETER (HELFILTER_PER_VIRTUAL_FLAVOR=%s)' % flag,
                          content)

        self.assertIn('INTEGER REAL_FLAVOR_INDEX_D(16)', content)
        self.assertIn('INTEGER BORN_FLAVOR_INDEX_D(16)', content)
        self.assertIn('INTEGER VIRTUAL_FLAVOR_INDEX_D(16)', content)
        self.assertIn('PARAMETER (MAX_VIRTUAL_FLAVOR_INDEX=16)', content)
        self.assertIn('PARAMETER (N_VIRTUAL_FLAVOR_CONFIGS=4)', content)
        self.assertIn('PARAMETER (HAS_PHYSICAL_FKS_CLASSES=.TRUE.)',
                      content)
        compact_content = ' '.join(content.lower().replace('$', '').split())
        compact_content = compact_content.replace(' ,', ',')
        self.assertIn('PARAMETER (NBORN_FLAVOR_CONFIGS=4)', content)
        self.assertIn('data born_fks_config_d / 1, 2, 3, 4 /',
                      content.lower())
        self.assertIn(
            'data born_fks_map_d / 1, 2, 3, 4, 1, 2, 3, 4, '
            '1, 2, 3, 4, 1, 2, 3, 4 /', compact_content)
        self.assertIn(
            'data fks_topology_d / 1, 1, 1, 1, 2, 2, 2, 2, '
            '3, 3, 3, 3, 4, 4, 4, 4 /', compact_content)
        self.assertIn('data real_flavor_index_d / 1, 2, 3, 4,',
                      content.lower())
        self.assertIn('data born_flavor_index_d / 1, 2, 3, 4,',
                      content.lower())
        self.assertNotIn(' 81,', content)
        self.assertNotIn('(-0.333', content)
        for index, info in enumerate(info_list, 1):
            self.assertIn(
                'DATA (BORN_PDG_TYPE_D(%d, IPOS), '
                'IPOS=1, NEXTERNAL-1) / %s /' %
                (index, ', '.join(str(pdg) for pdg in
                                  info['flavor_class']['born_pdgs'])),
                compact_content.upper())

        real_me = matrix_element.real_processes[0].matrix_element
        pdf_vars, pdf_data, pdf_lines, ee_vars = \
            exporter.get_pdf_lines_mir(real_me, 2)
        self.assertIn("INCLUDE 'fks_info.inc'", pdf_vars)
        self.assertIn('PDG_TYPE_D(NFKSPROCESS,1)', pdf_lines)
        self.assertIn('PDG_TYPE_D(NFKSPROCESS,2)', pdf_lines)
        self.assertIn('PD(IPROC)=FKS_PDF1*FKS_PDF2', pdf_lines)
        self.assertNotIn('_QUARK', pdf_vars + pdf_data + pdf_lines + ee_vars)

        _, _, mirrored_pdf_lines, _ = exporter.get_grouped_pdf_lines_mir(
            mirror=True)
        self.assertIn(
            'FKS_PDF_PDG1=PDG_TYPE_D(NFKSPROCESS,1)',
            mirrored_pdf_lines)
        self.assertIn('FKS_PDF1=PDG2PDF(LPP(2),', mirrored_pdf_lines)
        self.assertIn('FKS_PDF2=PDG2PDF(LPP(1),', mirrored_pdf_lines)

        handle, path = tempfile.mkstemp(
            prefix='leshouche_info_', suffix='.dat')
        os.close(handle)
        try:
            exporter.write_leshouche_info_file(path, matrix_element)
            with open(path) as stream:
                idup_rows = [tuple(int(pdg) for pdg in line.split()[3:])
                             for line in stream if line.startswith('I')]
        finally:
            if os.path.exists(path):
                os.remove(path)
        self.assertEqual(idup_rows,
                         [tuple(info['pdgs']) for info in info_list])
        self.assertTrue(all(abs(pdg) not in
                            interface._curr_model.get('merged_particles')
                            for row in idup_rows for pdg in row))
class TestRealChannelExport(unittest.TestCase):
    """Check real topology ownership and the helicity-summed partition."""

    @staticmethod
    def real_me(vertices):
        class MatrixElement(dict):
            def get_number_of_amplitudes(self):
                return len(vertices) + 2

        diagrams = [{'vertices': [{'legs': [None] * size} for size in sizes]}
                    for sizes in vertices]
        return SimpleNamespace(matrix_element=MatrixElement(
            base_amplitude={'diagrams': diagrams},
            diagrams=[{'number': number + 1} for number in range(len(vertices))]))

    def test_channels_own_sectors_without_born_channel_factor(self):
        # The contact diagram leaves a hole in diagram IDs, but not configs.
        matrix_element = SimpleNamespace(
            real_processes=[self.real_me([[3, 3], [4], [3, 3]]),
                            self.real_me([[3]])],
            get_fks_info_list=lambda: [{'n_me': 1}, {'n_me': 1}, {'n_me': 2}])
        exporter = object.__new__(export_fks.ProcessExporterFortranFKS)
        channels, sectors, representatives = exporter.get_real_channels(matrix_element)
        self.assertEqual(channels, [(1, 1, 1, 1), (1, 1, 2, 3), (2, 3, 1, 1)])
        self.assertEqual(sectors, [1, 1, 2])
        self.assertEqual(representatives, [1, 3])
        with tempfile.TemporaryDirectory() as path:
            filename = os.path.join(path, 'real_channels.inc')
            exporter.write_real_channels_file(writers.FortranWriter(filename),
                                              matrix_element)
            with open(filename) as stream:
                output = stream.read().lower()
            self.assertIn('n_real_channels=3', output)
            self.assertIn('max_real_diagrams=5', output)

    def test_no_real_channels(self):
        exporter = object.__new__(export_fks.ProcessExporterFortranFKS)
        matrix_element = SimpleNamespace(real_processes=[],
                                         get_fks_info_list=lambda: [])
        with tempfile.TemporaryDirectory() as path:
            filename = os.path.join(path, 'real_channels.inc')
            exporter.write_real_channels_file(writers.FortranWriter(filename),
                                              matrix_element)
            with open(filename) as stream:
                output = stream.read().lower()
            self.assertIn('n_real_channels=0', output)
            self.assertIn('real_channel_me(1)', output)
            self.assertIn('real_sector_me(1)', output)

    def test_channels_partition_each_physical_real_flavour(self):
        pairs = [(1, 2), (1, 7), (1, 2), (2, 2), (1, 7)]
        matrix_element = SimpleNamespace(
            real_processes=[self.real_me([[3, 3], [4], [3, 3]]),
                            self.real_me([[3]]), self.real_me([[3]])],
            get_fks_info_list=lambda: [
                {'n_me': ime, 'flavor_class': {'real_flavor_index': row}}
                for ime, row in pairs])
        exporter = object.__new__(export_fks.ProcessExporterFortranFKS)
        channels, sectors, representatives = exporter.get_real_channels(matrix_element)
        self.assertEqual(channels, [
            (1, 1, 1, 1), (1, 1, 2, 3),
            (1, 2, 1, 1), (1, 2, 2, 3), (2, 4, 1, 1)])
        self.assertEqual(sectors, [1, 1, 1, 2, 1])
        # A structural real ME with no executable physical class has no hard
        # channels; it must not make grouped mixed-order export fail.
        self.assertEqual(representatives, [1, 4, 0])

    def test_real_diagram_weights_follow_computed_helicities(self):
        compiler = shutil.which('gfortran')
        if not compiler:
            self.skipTest('gfortran is needed for the real helicity regression')
        filename = os.path.join(export_fks._file_path,
                  'iolibs/template_files/realmatrix_splitorders_fks.inc')
        aloha_functions = os.path.join(export_fks._file_path, os.pardir,
                          'aloha/template_files/aloha_functions.f')
        with open(filename) as stream:
            template = stream.read()
        template = template[template.index('      SUBROUTINE SMATRIX%(proc_prefix)s_SPLITORDERS'):]
        template = template[:template.index('C Helper functions')]
        # This test exercises the actual scalar filter and colour routine.
        # The independent vector evaluator has its own acceptance coverage.
        first = template.index('      RECURSIVE SUBROUTINE SMATRIX%(proc_prefix)s_SPLITORDERS_VEC')
        last = template.index('      BLOCK DATA HELFILTER%(proc_prefix)s')
        template = template[:first] + template[last:]
        template = template[:template.index('      SUBROUTINE MATRIX_%(proc_prefix)s_VEC')]

        # Helicity -1 has the same full matrix element as +1 and is not
        # recomputed; helicity 0 has none (its two diagrams cancel) and is
        # skipped. Both have nonzero diagram squares, which the weights
        # leave out at every point, the first one included.
        source = template % {
            'proc_prefix': '1', 'info_lines': '', 'process_lines': '',
            'hel_configs': 1, 'flavor_mask_decl': '', 'flavor_mask_setup': '',
            'helas_calls_vec': '', 'hel_configs_helas': 1, 'helas_flavor_mask_setup': '',
            'hel_index_setup': '      HEL_INDEX=1',
            'flavor_den_factor_setup': '',

            'ncomb': 3, 'nSqAmpSplitOrders': 1, 'nAmpSplitOrders': 1,
            'helicity_lines': '      data nhel /1,-1,0/',
            'den_factor_line': '      data iden /1/',
            'ngraphs': 2, 'nwavefuncs': 1, 'ncolor': 1,
            'wavefunctionsize': 8, 'nb_temp_jamp': 1,
            'color_data_lines': '      data cf /1,0/\n      data denom /1/',
            'helas_calls': ('      amp(1)=dcmplx(p(0,1)+nhel(1),0d0)\n'
                            '      amp(2)=dcmplx((2*nhel(1)**2-1)*p(0,1)'
                            '-nhel(1),0d0)'),
            'real_amp2_lines': ('      real_amp2(1)=real_amp2(1)+abs(amp(1))**2\n'
                               '      real_amp2(2)=real_amp2(2)+abs(amp(2))**2'),
            'jamp_lines': '      jamp(1,1)=amp(1)+amp(2)'}
        source += """
      integer function sqsoindex1(i,j)
      integer i,j
      sqsoindex1=1
      end
      program test_real_weights
      implicit none
      integer nfksprocess
      common /c_nfksprocess/nfksprocess
      double precision p(0:3,1),ans(0:1)
      include 'real_matrix_weights.inc'
      nfksprocess=1
      p=0d0
      get_real_diagram_weights=.true.
      real_amp2=-1d0
      p(0,1)=2d0
      call smatrix1_splitorders(p,ans)
      if (ans(0).ne.32d0.or.any(real_amp2.ne.(/9d0,1d0/))) stop 1
      p(0,1)=3d0
      call smatrix1_splitorders(p,ans)
      if (ans(0).ne.72d0.or.any(real_amp2.ne.(/16d0,4d0/))) stop 2
      get_real_diagram_weights=.false.
      real_amp2=-1d0
      p(0,1)=4d0
      call smatrix1_splitorders(p,ans)
      if (ans(0).ne.128d0.or.any(real_amp2.ne.-1d0)) stop 3
      end
"""
        with tempfile.TemporaryDirectory() as path:
            for name, content in {
                'model.f90': '''module model_object
type coupptr
complex(8), pointer :: p
end type
type flv_coupling
integer :: partner(1),partner2(1)
type(coupptr) :: val(1)
end type
end module
''',
                'matrix.f': source,
                'nexternal.inc': '      integer nexternal\n      parameter(nexternal=1)\n',
                'coupl.inc': '',
                'nFKSconfigs.inc': '',
                'fks_info.inc': ('      integer REAL_FLAVOR_INDEX_D(1)\n'
                                 '      parameter(REAL_FLAVOR_INDEX_D=(/1/))\n'),
                'real_matrix_weights.inc': (
                    '      double precision real_amp2(2)\n'
                    '      common /c_real_diagram_weights/ real_amp2\n'
                    '      logical get_real_diagram_weights\n'
                    '      common /c_get_real_diagram_weights/ get_real_diagram_weights\n')
            }.items():
                with open(os.path.join(path, name), 'w') as stream:
                    stream.write(content)
            result = subprocess.run([compiler, '-ffixed-line-length-none',
                                     '-fcheck=all', 'model.f90', aloha_functions,
                                     'matrix.f', '-o', 'check'],
                                    cwd=path, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            subprocess.check_output([os.path.join(path, 'check')], cwd=path,
                                    stderr=subprocess.STDOUT)


class TestBornGscale(unittest.TestCase):
    """gscale: when only alpha_S changed, born.f rescales its cached Born
    amplitudes by (G'/G)**n instead of recomputing them (BORN_GSCALE). With
    several split orders each one has its own power n, which must follow the
    QCD order of the amplitudes, and each coupling is checked against the
    QCD order of its interaction."""

    @classmethod
    def setUpClass(cls):
        interface = MGCmd.MasterCmd()
        interface.do_set('apply_flavor_grouping False --no_save')
        interface.do_import('model loop_sm')
        # s-channel gluon (QCD order 2) and photon/Z (QCD order 0)
        interface.do_generate('u u~ > t t~ QCD^2<=4 QED^2<=4 [real=QCD]')
        helas = fks_helas.FKSHelasMultiProcess(interface._fks_multi_proc)
        cls.born_me = helas.get_matrix_elements()[0].born_me
        cls.model = interface._curr_model

    @staticmethod
    def data(lines, name):
        """The values of the Fortran array NAME in the DATA lines"""
        values = []
        for m in re.finditer(r'DATA \(%s\(I\),I=\d+,\d+\) /([-0-9,]+)/'
                             % name, lines):
            values += [int(v) for v in m.group(1).split(',')]
        return values

    def test_mixed_split_order_powers(self):
        """Each amplitude, amplitude split order and squared split order
        gets the power of G of its own QCD order."""

        out = export_fks.ProcessExporterFortranFKS.get_born_gscale_data(
                                                                self.born_me)
        lines = out['gscale_data_lines']
        amp_qcd = self.data(lines, 'AMPQCD')
        self.assertEqual(sorted(amp_qcd), [0, 0, 2])
        # the amplitude split orders are sorted by weight: QCD first
        squared_orders, amp_orders = self.born_me.get_split_orders_mapping()
        self.assertEqual(self.data(lines, 'AMPSOQCD'), [2, 0])
        for (orders, numbers), power in zip(amp_orders, [2, 0]):
            self.assertEqual(set(amp_qcd[n - 1] for n in numbers), {power})
        # QCD^2=4, QCD^2=2 QED^2=2 and QED^2=4
        self.assertEqual(self.data(lines, 'SQSOQCD'), [4, 2, 0])
        self.assertIn('DATA GSCALE_FAST /.TRUE./', lines)
        self.assertEqual(out['maxqcdpow'], 4)

    def test_coupling_powers(self):
        """The coupling of the gluon to the quarks must scale as G, the
        electroweak ones not at all."""

        out = export_fks.ProcessExporterFortranFKS.get_born_coupling_check_dict(
                                                                self.born_me)
        self.assertEqual(out['born_checks_amps'], '.TRUE.')
        self.assertEqual(out['born_gscale'], '.TRUE.')
        gluon = set()
        for inter in self.model.get('interactions'):
            pdgs = sorted(p.get_pdg_code() for p in inter.get('particles'))
            if pdgs in ([-6, 6, 21], [-2, 2, 21]):
                gluon.update(inter.get('couplings').values())
        powers = dict(re.findall(r'abs\((\w+)-saved_coup\(\d+\)\*rpow\((\d)\)',
                                 out['coup_gscale_lines']))
        self.assertTrue(gluon & set(powers))
        for name, power in powers.items():
            self.assertEqual(power, '1' if name in gluon else '0', name)


class IOExportFKSTest(IOTests.IOTestManager):
    """Test class for the export fks module"""


    def generate(self, process, model, multiparticles=[]):
        """Create a process"""

        def run_cmd(cmd):
            opt = dict(interface.options)
            opt['ninja'] = None
            opt['collier'] = None
            opt['ninjars'] = None
            with misc.TMP_variable(interface, 'options', opt):
                interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        # The archived exporter fixtures describe grouped physical-flavour tables.
        interface._experimental_nlo_flavor_grouping = True
        
        if model.endswith('CMS'):
            run_cmd('set complex_mass_scheme')
            model = model[:-3]

        run_cmd('import model %s' % model)
        for multi in multiparticles:
            run_cmd('define %s' % multi)
        if isinstance(process, str):
            run_cmd('generate %s' % process)
        else:
            for p in process:
                run_cmd('add process %s' % p)

        files.rm(self.IOpath)
        run_cmd('output %s -f' % self.IOpath)


    @IOTests.createIOTest()
    def testIO_test_pptt_fksreal(self):
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]
        Template files identical in every subprocess directory: compare one.
        once: SubProcesses/[P0\w*/(sborn_sf\.f|maxparticles\.inc|has_ewsudakov\.inc|extra_cnt_wrapper\.f|ewsudakov_haslo\.inc|rescale_alpha_tagged\.f|boost_to_frame\.f|a0Gmuconv\.inc|numder_born\.f)$]"""
        self.generate(['p p > t t~ QED^2=0 QCD^2=4 [real=QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_ppw_fksall(self):
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]
        Template files identical in every subprocess directory: compare one.
        once: SubProcesses/[P0\w*/(sborn_sf\.f|maxparticles\.inc|has_ewsudakov\.inc|extra_cnt_wrapper\.f|ewsudakov_haslo\.inc|rescale_alpha_tagged\.f|boost_to_frame\.f|a0Gmuconv\.inc|numder_born\.f)$]"""
        self.generate(['p p > w+ QED^2=2 QCD^2=0 [QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_tdecay_fksreal(self):
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]
        Template files identical in every subprocess directory: compare one.
        once: SubProcesses/[P0\w*/(sborn_sf\.f|maxparticles\.inc|has_ewsudakov\.inc|extra_cnt_wrapper\.f|ewsudakov_haslo\.inc|rescale_alpha_tagged\.f|boost_to_frame\.f|a0Gmuconv\.inc|numder_born\.f)$]"""
        self.generate(['t > j j b QED^2=4 QCD^2=0 [real=QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_pptt_fks_loonly(self):
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]
        Template files identical in every subprocess directory: compare one.
        once: SubProcesses/[P0\w*/(sborn_sf\.f|maxparticles\.inc|has_ewsudakov\.inc|extra_cnt_wrapper\.f|ewsudakov_haslo\.inc|rescale_alpha_tagged\.f|boost_to_frame\.f|a0Gmuconv\.inc|numder_born\.f)$]"""
        self.generate(['p p > t t~ QED^2=0 QCD^2=4 [LOonly=QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_wprod_fksew(self):
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]
        Template files identical in every subprocess directory: compare one.
        once: SubProcesses/[P0\w*/(sborn_sf\.f|maxparticles\.inc|has_ewsudakov\.inc|extra_cnt_wrapper\.f|ewsudakov_haslo\.inc|rescale_alpha_tagged\.f|boost_to_frame\.f|a0Gmuconv\.inc|numder_born\.f)$]"""
        self.generate(['p p > e+ ve QED^2=4 QCD^2=0 [QED]'], 'loop_qcd_qed_smCMS')

    @IOTests.createIOTest()
    def testIO_test_pptt_fksrealew(self):
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]
        Template files identical in every subprocess directory: compare one.
        once: SubProcesses/[P0\w*/(sborn_sf\.f|maxparticles\.inc|has_ewsudakov\.inc|extra_cnt_wrapper\.f|ewsudakov_haslo\.inc|rescale_alpha_tagged\.f|boost_to_frame\.f|a0Gmuconv\.inc|numder_born\.f)$]"""
        self.generate(['p p > t t~ QED^2=0 QCD^2=4 [real=QED]'], 'sm', 
                      multiparticles = ['p = u u~ d d~ s s~ c c~ g a'])



class TestFKSOutput(unittest.TestCase):
    """ this class is to test that the new and old nlo generation give
    identical results
    """

    def test_fks_wavefunction_calls_exist_without_loops(self):
        """fks_singular.f only calls wavefunction routines, with the
        ALOHA-object interface, that the plain (non-loop) ALOHA library
        provides: real-only outputs such as [real=QCD QED] get that library,
        and an unknown routine made them fail to link."""
        import re
        with open(os.path.join(MGCmd.MG5DIR, 'Template', 'NLO',
                               'SubProcesses', 'fks_singular.f')) as stream:
            source = stream.read()
        with open(os.path.join(MGCmd.MG5DIR, 'aloha', 'template_files',
                               'aloha_functions.f')) as stream:
            library = stream.read().lower()
        calls = re.findall(r'call\s+([iosv]xxx\w\w)\s*\(([^)]*)\)',
                           source, re.I)
        self.assertTrue(calls)
        for name, arguments in calls:
            self.assertIn('subroutine %s(' % name.lower(), library, name)
            # p, mass, helicity, particle/antiparticle, flavor, output
            self.assertEqual(len(arguments.split(',')), 6, name)

    def test_add_wgt_calls_pass_all_arguments(self):
        """Native weights take five arguments; explicit buffers take seven.

        Both APIs require an orders array, including artificial soft weights.
        """
        import glob
        import re

        def arguments(source, start):
            # count the top-level arguments of the parenthesis at 'start'
            depth, count = 0, 1
            for char in source[start:]:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth == 0:
                        return count
                elif char == ',' and depth == 1:
                    count += 1
            raise AssertionError('unbalanced call')

        template = os.path.join(MGCmd.MG5DIR, 'Template', 'NLO',
                                'SubProcesses')
        arities = {}
        calls = []
        for path in glob.glob(os.path.join(template, '*.f')):
            with open(path) as stream:
                # join Fortran fixed-form continuation lines
                source = re.sub(r'\n     [$&]', '', stream.read())
            for match in re.finditer(r'subroutine\s+(add_wgt(?:_buffered)?)\s*\(',
                                     source, re.I):
                arities[match[1].lower()] = arguments(source, match.end() - 1)
            for match in re.finditer(r'call\s+(add_wgt(?:_buffered)?)\s*\(', source, re.I):
                calls.append((os.path.basename(path),
                              match[1].lower(),
                              arguments(source, match.end() - 1),
                              source[match.start():match.start() + 60]))
        self.assertEqual(arities, {'add_wgt': 5, 'add_wgt_buffered': 7})
        self.assertTrue(calls)
        for name, routine, count, text in calls:
            self.assertEqual(count, arities[routine], '%s: %s' % (name, text))

    def test_bornsoftvirtual_recomputes_link_borns(self):
        """The integrated soft term of bornsoftvirtual evaluates the
        colour/charge links for nFKSprocess_col_used/_chg_used, whose Born
        counterterm values (ans_cnt, split orders) differ from those of the
        configuration passed in. It must recompute them there, as MG5 does
        with sborn: reusing the passed born_cnt changed the integrated soft
        term at almost every phase-space point (W+j, mixed QCD/QED), while
        with the recomputation the integrand equals MG5's point by point."""
        import re
        with open(os.path.join(MGCmd.MG5DIR, 'Template', 'NLO',
                               'SubProcesses', 'fks_buffered.f')) as stream:
            source = stream.read()
        start = source.index('      subroutine bornsoftvirtual_buffered(')
        body = source[start:source.index('\n      end\n', start)]
        # the link loop: from the I(reg) comment to restoring nFKSprocess
        begin = body.index('I(reg) terms')
        stop = body.index('nFKSprocess = nFKSprocess_save\n', begin)
        loop = body[begin:stop]
        code = '\n'.join(line for line in loop.splitlines()
                         if not re.match(r'[cC*!]', line))
        chooser = code.index('call fks_inc_chooser()')
        recompute = code.find('call sborn_amp_frame(p_born', chooser)
        links = code.index('call sborn_sf_store(', chooser)
        self.assertNotEqual(recompute, -1,
                            'the link loop must recompute the Born')
        self.assertLess(recompute, links)
        # and the configuration's own Born values are restored afterwards
        after = body[stop:]
        self.assertIn('ans_cnt(:,:) = born_cnt(:,:)', after)

    def test_tir_library_paths_are_available_at_runtime(self):
        """External TIR shared libraries need an rpath in NLO executables."""
        exporter = object.__new__(export_fks.ProcessExporterFortranFKS)
        exporter.mgme_dir = MGCmd.MG5DIR
        make_opts = exporter.write_make_opts(None, [
            '-L/opt/ninja/lib/ -lninja',
            '-L/opt/ninja/lib/ -lavh_olo',
            '-L/opt/collier/ -lcollier'], [])

        self.assertIn('-Wl,-rpath,/opt/ninja/lib/', make_opts)
        self.assertIn('-Wl,-rpath,/opt/collier/', make_opts)
        self.assertEqual(
            make_opts.count('-Wl,-rpath,/opt/ninja/lib/'), 1)

    def tearDown(self):
        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        run_cmd('set low_mem_multicore_nlo_generation False')
        run_cmd('set OLP MadLoop')

    # what differs between two outputs of the same process: the dates, the
    # timings, the command history (which has the option itself) and the
    # pickled model (a copy of the model of the session)
    # gen_jpeg-pl copies each subprocess thumbnail to this shared preview;
    # its final image depends on worker completion order, unlike the sources.
    ignored_files = ['amcatnlo.tar.gz', 'index.html', os.path.join('HTML', 'card.jpg'),
                     os.path.join('Cards', 'proc_card_mg5.dat')]
    ignored_patterns = [r'_debug\.log$', r'\.pkl$']

    def output_both_modes(self, commands, name='PROC', output_options=''):
        """Run the commands (the generation) and the output, with the
        low_mem_multicore_nlo_generation mode and then without it. Returns
        the two output directories, in this order (they have the same name,
        in different directories: {dir} in the commands), written with the
        output_options"""
        import multiprocessing
        path = tempfile.mkdtemp('', 'TMPLowMemTest', None)
        self.addCleanup(shutil.rmtree, path, True)

        def output(mode):
            def run_cmd(cmd):
                interface.exec_cmd(cmd, errorhandling=False, printcmd=False,
                                   precmd=True, postcmd=True)
            interface = MGCmd.MasterCmd()
            run_cmd('set low_mem_multicore_nlo_generation %s' % (mode == 'low_mem'))
            for command in commands:
                run_cmd(command.replace('{dir}', os.path.join(path, mode)))
            run_cmd('output %s %s' % (os.path.join(path, mode, name), output_options))

        outputs = []
        for mode in ['low_mem', 'serial']:
            os.mkdir(os.path.join(path, mode))
            # each in a process of its own, forked from this one, so that both
            # start from the same state: a session keeps some traces of what it
            # did (e.g. the merged Lorentz structures of the model, whose names
            # are numbered). One after the other: the first output of a fresh
            # installation also downloads the model and compiles CutTools and
            # IREGI in it, which two outputs would do at the same time.
            process = multiprocessing.get_context('fork').Process(
                                                  target=output, args=(mode,))
            process.start()
            process.join()
            self.assertEqual(process.exitcode, 0,
                             'the output failed in the %s mode' % mode)
            outputs.append(os.path.join(path, mode, name))
        return outputs

    def assert_same_output(self, new, old):
        """The two output directories have the same directories, links and
        files, with the same content"""
        def tree(top):
            entries = {}
            for dirpath, dirnames, filenames in os.walk(top):
                for name in dirnames + filenames:
                    path = os.path.join(dirpath, name)
                    rel = os.path.relpath(path, top)
                    if rel in self.ignored_files or \
                       any(re.search(p, rel) for p in self.ignored_patterns):
                        continue
                    if os.path.islink(path):
                        entries[rel] = ('link', os.readlink(path))
                    elif os.path.isdir(path):
                        entries[rel] = ('dir', None)
                    else:
                        entries[rel] = ('file', path)
            return entries

        new_tree = tree(new)
        old_tree = tree(old)
        self.assertEqual(sorted(new_tree), sorted(old_tree))
        different = []
        for rel in sorted(old_tree):
            (kind, value), (new_kind, new_value) = old_tree[rel], new_tree[rel]
            if kind != new_kind or \
               (kind == 'link' and value != new_value) or \
               (kind == 'file' and not filecmp.cmp(value, new_value, shallow=False)):
                different.append(rel)
        if different:
            import difflib
            diffs = []
            for rel in different[:3]:
                if old_tree[rel][0] == new_tree[rel][0] == 'file':
                    with open(old_tree[rel][1], errors='replace') as stream:
                        old_lines = stream.readlines()
                    with open(new_tree[rel][1], errors='replace') as stream:
                        new_lines = stream.readlines()
                    diffs.extend(list(difflib.unified_diff(old_lines, new_lines,
                                    'serial/' + rel, 'low_mem/' + rel))[:40])
            self.fail('%s differ:\n%s' % (', '.join(different), ''.join(diffs)))

    def test_w_nlo_gen_qcd(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results for p p > e+ ve [QCD], including
        the private DP terminal updaters of the loop directories, which the
        processes writing the directories leave for the parent to prove
        """
        outputs = self.output_both_modes(
                ['generate p p > e+ ve QED^2=4 QCD^2=0 [QCD]'], 'W')
        for output in outputs:
            batches = glob.glob(os.path.join(output, 'SubProcesses', 'P*', 'V*',
                                             'coef_construction_*.f'))
            self.assertTrue(batches)
            for batch in batches:
                with open(batch) as stream:
                    self.assertIn('CALL DP_TERMINAL_WL_1_1(', stream.read(), batch)
        self.assert_same_output(*outputs)


#    @test_manager.bypass_for_py3
    def test_w_nlo_gen_qed(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results for p p > e+ ve [QED]
        """
        self.assert_same_output(*self.output_both_modes(
                ['generate p p > e+ ve QED^2=4 QCD^2=0 [QED]'], 'W'))


 #   @test_manager.bypass_for_py3
    def test_z_nlo_gen_qed(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results for p p > e+ e- [QED], in particular
        that b-initiated processes are NOT combined with d-s ones (they have
        the born of the d-s ones, not their virtual: they are combined in a
        second round, after the photon-initiated one, but the matrix elements
        must stay in the order of the serial mode). Also that both modes give
        the same uid to each matrix element: its position, from 1.
        Process.shell_string ends the names longer than 64 characters with it,
        and here with it always: the b-initiated matrix elements must not get
        the uid of another one.
        """
        from unittest import mock
        shell_string = base_objects.Process.shell_string
        def shell_string_with_uid(proc, *args, **opts):
            name = shell_string(proc, *args, **opts)
            main = opts.get('main', args[2] if len(args) > 2 else True)
            if main and not name.endswith('_%d' % proc['uid']):
                name += '_%d' % proc['uid']
            return name
        with mock.patch.object(base_objects.Process, 'shell_string',
                               shell_string_with_uid):
            new, old = self.output_both_modes(
                    ['define p3 = d s b d~ s~ b~ a',
                     'generate p3 p3 > e+ e- QED^2=4 QCD^2=0 [QED]'], 'Z')
        with open(os.path.join(old, 'SubProcesses', 'subproc.mg')) as stream:
            names = [line.strip() for line in stream]
        self.assertEqual([name.rsplit('_', 1)[1] for name in names],
                         [str(uid) for uid in range(1, len(names) + 1)])
        self.assertTrue([name for name in names if name.startswith('P0_bbx_emep_')])
        self.assert_same_output(new, old)

#    @test_manager.bypass_for_py3
    def test_z_nlo_gen_qcd(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results for p p > e+ e- [QCD]
        """
        self.assert_same_output(*self.output_both_modes(
                ['define p3 = d s b d~ s~ b~',
                 'generate p3 p3 > e+ e- QED^2=4 QCD^2=0 [QCD]'], 'Z'))

#    @test_manager.bypass_for_py3
    def test_wj_loonly_gen(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results for p p > w j [LOonly=QCD], in
        particular the initial states of the borns (there are no reals)
        """
        new, old = self.output_both_modes(['generate p p > w+ j [LOonly=QCD]'], 'W')
        with open(os.path.join(new, 'SubProcesses', 'initial_states_map.dat')) as stream:
            self.assertTrue(stream.read().strip())
        # an [LOonly] directory gets its trivial real_from_born_configs.inc
        # (the borns, as the reals of the fake FKS configuration)
        for path in glob.glob(os.path.join(new, 'SubProcesses', 'P0_*',
                                           'real_from_born_configs.inc')):
            with open(path) as stream:
                self.assertIn('DATA (REAL_FROM_BORN_CONF(IRFBC,1)', stream.read())
        self.assert_same_output(new, old)

    def test_add_process_nlo_gen(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results when a process with the same
        multiplicity is added: fks_j_from_i is found from the borns of the
        process definition of each real only, so that u g > t t~ u and
        d g > t t~ d stay combined in the g g > t t~ directory
        """
        self.assert_same_output(*self.output_both_modes(
                ['generate g g > t t~ [QCD]',
                 'add process u u~ > t t~ [QCD]'], 'TT'))

    def test_nlo_mixed_expansion_false_gen(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same virtual with nlo_mixed_expansion False:
        the orders of the loops are those of the borns (no EW loop)
        """
        self.assert_same_output(*self.output_both_modes(
                ['import model loop_qcd_qed_sm',
                 'set nlo_mixed_expansion False',
                 'generate u u~ > w+ w- [QCD]'], 'WW'))
        
    def test_splitting_types_nlo_gen(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results when only some matrix elements have
        QED splittings (here g g > t t~, not g a > t t~ and a g > t t~ which
        come after it), and that real_from_born_configs.inc of a directory
        has the mapping of the borns to the reals of each of its FKS
        configurations, QCD or QED, whatever the other ones and the other
        directories (for the H events, also with --allow_qed_ps)
        """
        new, old = self.output_both_modes(
                ['import model loop_qcd_qed_sm',
                 'define p = g a',
                 'generate p p > t t~ [QCD QED]'], 'TT')
        with open(os.path.join(old, 'SubProcesses', 'subproc.mg')) as stream:
            self.assertEqual([line.strip() for line in stream],
                             ['P0_gg_ttx', 'P0_ga_ttx', 'P0_ag_ttx'])
        for directory in ['P0_gg_ttx', 'P0_ga_ttx', 'P0_ag_ttx']:
            path = os.path.join(old, 'SubProcesses', directory)
            # the FKS configurations of the directory and those of the others
            # which its MC histories reach, all in real_from_born_configs.inc
            with open(os.path.join(path, 'nFKSconfigs.inc')) as stream:
                nconfigs = int(re.search(r'FKS_CONFIGS\s*=\s*(\d+)',
                                         stream.read(), re.I).group(1))
            with open(os.path.join(path, 'real_from_born_configs.inc')) as stream:
                mapped = [int(config) for config in re.findall(
                              r'REAL_FROM_BORN_CONF\(IRFBC,(\d+)\)',
                              stream.read(), re.I)]
            self.assertEqual(mapped, list(range(1, nconfigs + 1)), directory)
        self.assert_same_output(new, old)

    def test_add_process_other_orders_nlo_gen(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results when two process definitions have
        a real process with other orders (g u > t t~ u, QCD^2=6 QED^2=2 for
        g g > t t~ and QCD^2=4 QED^2=4 for g a > t t~): each born has the real
        matrix element of its own definition
        """
        new, old = self.output_both_modes(
                ['import model loop_qcd_qed_sm',
                 'generate g g > t t~ [real=QCD QED]',
                 'add process g a > t t~ [real=QCD QED]'], 'TT')
        reals = ''
        for path in glob.glob(os.path.join(old, 'SubProcesses', 'P0_ga_ttx',
                                           'matrix_*.f')):
            with open(path) as stream:
                reals += stream.read()
        self.assertIn('g u > t t~ u [ real = QCD QED ] QCD^2=4 QED^2=4', reals)
        self.assertNotIn('g u > t t~ u [ real = QCD QED ] QCD^2=6', reals)
        self.assert_same_output(new, old)

    def test_reals_of_each_definition_nlo_gen(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results when two process definitions have a
        real process in common, in different NLO modes (u g > t t~ u, of
        g g > t t~ [QCD] and of u u~ > t t~ [real=QCD]): each born has the
        real matrix element of its own definition"""
        new, old = self.output_both_modes(
                ['generate g g > t t~ [QCD]',
                 'add process u u~ > t t~ [real=QCD]'], 'TT')
        reals = ''
        for path in glob.glob(os.path.join(old, 'SubProcesses', 'P0_uux_ttx',
                                           'matrix_*.f')):
            with open(path) as stream:
                reals += stream.read()
        self.assertIn('[ real = QCD', reals)
        self.assertNotIn('[ all = QCD', reals)
        self.assert_same_output(new, old)

    def test_loop_filter_of_added_process_nlo_gen(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results when a process added later has a
        --loop_filter: the virtual of each born is generated with the filter
        of its process definition (the filter of d d~ > e+ e- removes all the
        loops of u u~ > e+ e-)"""
        new, old = self.output_both_modes(
                ['import model loop_sm',
                 'generate u u~ > e+ e- [QCD]',
                 'add process d d~ > e+ e- [QCD] --loop_filter="not 2 in loop_pdgs"'],
                'Z')
        for directory in ['P0_uux_emep', 'P0_ddx_emep']:
            self.assertTrue(glob.glob(os.path.join(old, 'SubProcesses',
                                                   directory, 'V*')), directory)
        self.assert_same_output(new, old)

    def test_output_after_add_process_nlo_gen(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results for an output after another one and
        an 'add process': the old mode, which keeps the matrix elements of an
        output for the next one, generates them again"""
        new, old = self.output_both_modes(
                ['generate u u~ > e+ e- [real=QCD]',
                 'output {dir}/FIRST',
                 'add process d d~ > e+ e- [real=QCD]'], 'Z')
        with open(os.path.join(old, 'SubProcesses', 'subproc.mg')) as stream:
            self.assertEqual([line.strip() for line in stream],
                             ['P0_uux_emep', 'P0_ddx_emep'])
        self.assert_same_output(new, old)

    def test_ewsudakov_added_process_nlo_gen(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results when a process with the EW Sudakov
        corrections is added to one without them: their matrix elements are
        not combined (c c~ > z z keeps its Sudakov), whichever comes first"""
        new, old = self.output_both_modes(
                ['import model loop_qcd_qed_sm_Gmu_forSudakov',
                 'generate u u~ > z z [LOonly=QCD]',
                 'add process c c~ > z z [LOonly=QCD] --ewsudakov'], 'ZZ')
        with open(os.path.join(old, 'SubProcesses', 'subproc.mg')) as stream:
            self.assertEqual([line.strip() for line in stream],
                             ['P0_uux_zz', 'P0_ccx_zz'])
        self.assert_same_output(new, old)

    def test_fks_standalone_info_page(self):
        """check that the new (memory and cpu efficient) and old generation
        mode give the same standalone FKS output (output --fks), whose info
        page lists all the P directories: it is made before they are stripped
        down to what the Born check needs"""
        new, old = self.output_both_modes(
                ['generate u u~ > e+ e- [real=QCD]',
                 'add process d d~ > e+ e- [real=QCD]'], 'Z', '--fks')
        with open(os.path.join(old, 'HTML', 'info.html')) as stream:
            info = stream.read()
        for directory in ['P0_uux_emep', 'P0_ddx_emep']:
            self.assertIn(directory, info)
            self.assertFalse(glob.glob(os.path.join(old, 'SubProcesses',
                                                    directory, 'matrix_*.f')))
        self.assert_same_output(new, old)

    def test_madanalysis5_card_processes(self):
        """the default MadAnalysis5 card of an NLO output gets the born
        processes of the matrix elements, also in the
        low_mem_multicore_nlo_generation mode (which has no matrix element
        any more when it finalizes the output)"""
        from unittest import mock
        import madgraph.iolibs.export_v4 as export_v4
        path = tempfile.mkdtemp('', 'TMPMA5Test', None)
        self.addCleanup(shutil.rmtree, path, True)
        processes = {}
        for low_mem in [True, False]:
            interface = MGCmd.MasterCmd()
            def run_cmd(cmd):
                interface.exec_cmd(cmd, errorhandling=False, printcmd=False,
                                   precmd=True, postcmd=True)
            run_cmd('set low_mem_multicore_nlo_generation %s' % low_mem)
            run_cmd('generate p p > e+ e- [real=QCD]')
            interface.options['madanalysis5_path'] = 'MA5'
            with mock.patch.object(export_v4.ProcessExporterFortran,
                                   'create_default_madanalysis5_cards') as cards:
                run_cmd('output %s' % os.path.join(path, str(low_mem)))
            processes[low_mem] = [process.nice_string() for process in \
                                  cards.call_args[0][2][0]]
        self.assertTrue(processes[False])
        self.assertEqual(processes[True], processes[False])

    def test_low_mem_failed_generation_keeps_the_output(self):
        """the low_mem_multicore_nlo_generation mode generates the real and
        virtual amplitudes at the output: before the output directory is
        replaced, which a failure leaves as it was"""
        from unittest import mock
        path = tempfile.mkdtemp('', 'TMPWTest', None)
        self.addCleanup(shutil.rmtree, path, True)

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        run_cmd('set low_mem_multicore_nlo_generation True')
        run_cmd('generate u u~ > e+ e- [real=QCD]')
        output = os.path.join(path, 'Z')
        run_cmd('output %s' % output)
        def fail(args):
            raise fks_common.FKSProcessError('the born failed')
        with mock.patch.object(fks_helas, 'async_generate_born', fail):
            self.assertRaises(fks_common.FKSProcessError, run_cmd,
                              'output %s -f' % output)
        self.assertTrue(os.path.exists(os.path.join(output, 'SubProcesses',
                                                    'subproc.mg')))
        self.assertNotIn('tmpdir', interface._curr_matrix_elements)
        run_cmd('output %s -f' % output)
        self.assertTrue(os.path.exists(os.path.join(output, 'SubProcesses',
                                                    'subproc.mg')))

    def test_polarised_default_frame_is_the_massive_final_state(self):
        """The partonic c.m., me_frame = [1, 2], has no infrared-safe meaning
        for p p > w+{0} j at NLO: with initial-state QCD radiation the frame
        the FKS momenta come in depends on the sector (+9.5(4)% when only
        fks_b changed), and the only frame equal to the partonic c.m. of the
        Born contains the jet. So the output must write the rest frame of the
        massive final state, [3], as the run_card default, and the run_card
        must still refuse [1, 2] when asked for explicitly."""
        import madgraph.various.banner as banner_mod
        path = tempfile.mkdtemp('', 'TMPWPolTest', None)
        interface = MGCmd.MasterCmd()
        try:
            interface.exec_cmd('generate p p > w+{0} j [real=QCD]',
                               errorhandling=False, printcmd=False,
                               precmd=True, postcmd=True)
            interface.exec_cmd('output %s' % os.path.join(path, 'W'),
                               errorhandling=False, printcmd=False,
                               precmd=True, postcmd=True)

            charac = banner_mod.ProcCharacteristic(os.path.join(
                path, 'W', 'SubProcesses', 'proc_characteristics'))
            self.assertEqual(charac['me_frame_cm_legs'], [3, 4])
            self.assertEqual(charac['me_frame_unsafe_legs'], [1, 2, 4])
            self.assertEqual(charac['me_frame_default_legs'], [3])

            # the card as written: the W rest frame, accepted at launch
            run_card = banner_mod.RunCardNLO(os.path.join(
                path, 'W', 'Cards', 'run_card.dat'))
            self.assertEqual(run_card['me_frame'], [3])
            run_card.set_me_frame_process(charac)
            run_card.check_validity()
            run_card.update_system_parameter_for_include()
            self.assertEqual(run_card['frame_id'], 8)
            # [1, 2] by hand: refused with real emission
            run_card.set('me_frame', [1, 2], user=True)
            self.assertRaises(banner_mod.InvalidRunCard,
                              run_card.check_validity)
            # the Born alone (launch LO) has nothing to be infrared safe for
            run_card.set_me_frame_process(charac, real_emission=False)
            run_card.check_validity()

            # A bare 'treatcards run', as the reweighting and the FKS
            # standalone check compile their Source: no run_card is loaded
            # in that shell. It used to die on self.run_card; now the include
            # file resolves me_frame against the process found on disk.
            me_dir = os.path.join(path, 'W')
            def treatcards():
                with open(os.path.join(me_dir, 'treat.cmd'), 'w') as cmd:
                    cmd.write('treatcards run\n')
                return subprocess.run(
                    [sys.executable, os.path.join('bin', 'aMCatNLO'),
                     'treat.cmd'], cwd=me_dir, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, universal_newlines=True).stdout
            def frame_id():
                with open(os.path.join(me_dir, 'Source', 'run_card.inc')) as inc:
                    return [l.split('=')[1].strip() for l in inc
                            if 'FRAME_ID' in l.upper()]
            out = treatcards()
            self.assertNotIn('Error', out)
            self.assertEqual(frame_id(), ['8'])
            # a beam that radiates is refused on that path too
            card_path = os.path.join(me_dir, 'Cards', 'run_card.dat')
            with open(card_path) as card:
                text = card.read()
            text = re.sub(r'^(\s*)3(\s*=\s*me_frame)', r'\g<1>1, 3\2', text,
                          count=1, flags=re.M)
            with open(card_path, 'w') as card:
                card.write(text)
            self.assertIn('radiate', treatcards())
        finally:
            shutil.rmtree(path, ignore_errors=True)


    def test_low_mem_output_twice(self):
        """a second output of the same generation in the
        low_mem_multicore_nlo_generation mode generates the matrix elements
        again: the files of the first one are removed with it. Only the worker
        processes load these files (the matrix elements), also for the records
        from which the exporter assigns the Born providers, before the
        directories are written: the parent process never holds them"""
        from unittest import mock
        path = tempfile.mkdtemp('', 'TMPWTest', None)
        self.addCleanup(shutil.rmtree, path, True)

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False,
                               precmd=True, postcmd=True)

        parent = os.getpid()
        load_tmpfile = fks_helas.load_tmpfile
        def load_in_worker(*args):
            if os.getpid() == parent:
                raise AssertionError('a file was loaded in the parent process')
            return load_tmpfile(*args)

        interface = MGCmd.MasterCmd()
        run_cmd('set low_mem_multicore_nlo_generation True')
        run_cmd('generate p p > e+ ve QED^2=4 QCD^2=0 [QCD]')
        subprocs = []
        for name in ['W1', 'W2']:
            with mock.patch.object(fks_helas, 'load_tmpfile', load_in_worker):
                run_cmd('output %s' % os.path.join(path, name))
            self.assertNotIn('tmpdir', interface._curr_matrix_elements)
            with open(os.path.join(path, name, 'SubProcesses', 'subproc.mg')) as stream:
                subprocs.append(stream.read())
            self.assertTrue(glob.glob(os.path.join(path, name, 'SubProcesses',
                                                   'P*', 'born_support.json')))
        self.assertTrue(subprocs[0])
        self.assertEqual(subprocs[0], subprocs[1])

    @test_manager.bypass_for_py3
    def test_w_nlo_gen_gosam(self):
        """check that the new generation mode works when gosam is set 
        for p p > w [QCD] 
        """
        path = tempfile.mkdtemp('', 'TMPWTest', None)

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        try:
            run_cmd('set low_mem_multicore_nlo_generation True')
            run_cmd('set OLP GoSam')
            run_cmd('generate p p > w+ QED^2=2 QCD^2=0 [QCD]')
            try:
                run_cmd('output %s' % os.path.join(path, 'W-newway'))
            except fks_common.FKSProcessError as err:
                # catch the error if gosam is not there
                if not 'Generation of the virtuals with GoSam failed' in str(err):
                    raise Exception(err)
        except Exception as e:
            run_cmd('set low_mem_multicore_nlo_generation False')
            run_cmd('set OLP MadLoop')
            raise e
        finally:
            run_cmd('set low_mem_multicore_nlo_generation False')
            run_cmd('set OLP MadLoop')

        shutil.rmtree(path)


class TestColorMatrixEncodingGate(unittest.TestCase):
    """The color matrix may be written in the compressed form -- one line per
    orbit of the color basis symmetry, the entries rebuilt on the first call by
    INIT_CF -- only by the templates that declare CF in the common block
    INIT_CF fills and call it.

    The FKS templates do none of that: CF is a plain local array whose only
    filling is the DATA statements of get_color_data_lines. Writing the
    compressed form there left CF at zero, so every real emission matrix
    element with a large enough color basis came out zero -- for
    g g > t t~ g g g (120 color structures) that is every soft and collinear
    check of test_soft_col_limits, which refuses to let the run start.
    """

    @classmethod
    def setUpClass(cls):
        # g g > g g: six color structures with a non-trivial symmetry, so the
        # compressed form is available whenever the exporter allows it
        interface = MGCmd.MasterCmd()
        interface.exec_cmd('import model sm', errorhandling=False,
                           printcmd=False, precmd=True, postcmd=True)
        interface.exec_cmd('generate g g > g g', errorhandling=False,
                           printcmd=False, precmd=True, postcmd=True)
        cls.matrix_element = helas_objects.HelasMatrixElement(
                                                        interface._curr_amps[0])

    @staticmethod
    def entry_lines(lines):
        """The DATA statements holding the color matrix entries themselves."""
        return [line for line in lines if 'CF(i)' in line or 'CF(i,' in line]

    def exporter(self, cls):
        """An exporter of that class with the compression always worth taking,
        so that only color_matrix_encoding_allowed decides."""
        instance = cls.__new__(cls)
        instance.color_encoding_margin = 0
        return instance

    def test_standalone_compresses(self):
        """The plain standalone output does rebuild CF at run time, so it only
        writes the denominator out. Without this the tests below would pass on
        a matrix element that is never compressed in the first place."""

        exporter = self.exporter(export_v4.ProcessExporterFortranSA)
        exporter.opt = {'export_format': 'standalone'}
        self.assertTrue(exporter.color_matrix_encoding_allowed(
                                                          self.matrix_element))
        lines = exporter.get_color_data_lines(self.matrix_element)
        self.assertEqual([], self.entry_lines(lines))
        self.assertTrue(any('Denom' in line for line in lines))

    def test_fks_writes_the_entries_out(self):
        """The FKS exporter has no INIT_CF, so the entries must be written."""

        exporter = self.exporter(export_fks.ProcessExporterFortranFKS)
        self.assertFalse(exporter.color_matrix_encoding_allowed(
                                                          self.matrix_element))
        lines = exporter.get_color_data_lines(self.matrix_element)
        self.assertNotEqual([], self.entry_lines(lines))

    def test_split_orders_and_madspin_write_the_entries_out(self):
        """The standalone variants select a template of their own -- split
        orders, MadSpin's msP/msF, matchbox -- and none of them rebuilds CF."""

        for export_format in ('standalone_msP', 'standalone_msF', 'matchbox',
                              'madloop_matchbox'):
            exporter = self.exporter(export_v4.ProcessExporterFortranSA)
            exporter.opt = {'export_format': export_format}
            self.assertFalse(exporter.color_matrix_encoding_allowed(
                                                           self.matrix_element),
                             'encoding allowed for %s' % export_format)

        # split orders send the plain standalone output to
        # matrix_standalone_splitOrders_v4.inc, which has no INIT_CF either
        split = copy.deepcopy(self.matrix_element)
        split.get('processes')[0].set('split_orders', ['QCD'])
        exporter = self.exporter(export_v4.ProcessExporterFortranSA)
        exporter.opt = {'export_format': 'standalone'}
        self.assertFalse(exporter.color_matrix_encoding_allowed(split))
        self.assertNotEqual([], self.entry_lines(
                                        exporter.get_color_data_lines(split)))

    def test_madevent_compresses_and_madweight_does_not(self):
        """madevent calls INIT_CF from every one of its templates; MadWeight
        writes a template of its own that does not."""

        exporter = self.exporter(export_v4.ProcessExporterFortranME)
        self.assertTrue(exporter.color_matrix_encoding_allowed(
                                                          self.matrix_element))

        exporter = self.exporter(export_v4.ProcessExporterFortranMW)
        self.assertFalse(exporter.color_matrix_encoding_allowed(
                                                          self.matrix_element))

    def test_whitelist_matches_the_templates(self):
        """COLOR_MATRIX_ENCODING_TEMPLATES is the list of templates that call
        INIT_CF. A template added or changed on either side has to move on the
        other, or an exporter is again free to drop the entries of a CF that
        nothing fills."""

        directory = os.path.join(os.path.dirname(export_v4.__file__),
                                 'template_files')
        seen = set()
        for name in os.listdir(directory):
            if not name.endswith('.inc'):
                continue
            text = open(os.path.join(directory, name)).read()
            if '%(color_data_lines)s' not in text:
                self.assertNotIn('%(color_init_routine)s', text,
                                 '%s rebuilds a color matrix it never writes'
                                 % name)
                continue
            rebuilds = 'INIT_CF' in text
            self.assertEqual(rebuilds,
                             name in export_v4.COLOR_MATRIX_ENCODING_TEMPLATES,
                             '%s: INIT_CF=%s but whitelisted=%s' % \
                             (name, rebuilds,
                              name in export_v4.COLOR_MATRIX_ENCODING_TEMPLATES))
            if rebuilds:
                self.assertIn('%(color_init_routine)s', text,
                              '%s calls INIT_CF but never defines it' % name)
                seen.add(name)
        self.assertEqual(seen, set(export_v4.COLOR_MATRIX_ENCODING_TEMPLATES))
