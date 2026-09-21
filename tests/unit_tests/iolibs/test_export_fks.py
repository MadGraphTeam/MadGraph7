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
import fractions
import os 
import re
import sys
import tempfile
import glob
import shutil
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
import madgraph.core.base_objects as base_objects
import madgraph.iolibs.export_fks as export_fks
import madgraph.iolibs.file_writers as file_writers
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

    def test_openmp_amplitude_loop_is_explicitly_opt_in(self):
        """The paper loop must be active in OpenMP builds but serial by
        default, rather than hidden behind the invalid historical ``!omp``
        comment sentinel.
        """

        template_root = os.path.join(
            root_path, os.path.pardir, os.path.pardir, 'Template', 'NLO')
        with open(os.path.join(
                template_root, 'SubProcesses', 'driver_mintMC.f')) as stream:
            driver = stream.read().upper()
        self.assertIn(
            '!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(IVEC)', driver)
        self.assertIn('!$OMP END PARALLEL DO', driver)
        self.assertNotIn('!OMP PARALLEL DO', driver)
        self.assertIn('WALL TIME IN VECTOR_AMPLITUDE', driver)

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

    def test_grouped_virtual_disables_shared_helicity_sampling(self):
        """Grouped virtual rows cannot consume one shared HelFilter.dat."""

        template_dir = os.path.join(
            root_path, os.path.pardir, os.path.pardir,
            'Template', 'NLO', 'SubProcesses')
        for filename in ('driver_mintFO.f', 'driver_mintMC.f'):
            with open(os.path.join(template_dir, filename)) as stream:
                content = ' '.join(stream.read().upper().split())
            self.assertIn(
                'I.EQ.0.OR.N_VIRTUAL_FLAVOR_CONFIGS.GT.1', content)

        with open(os.path.join(template_dir, 'BinothLHA.f')) as stream:
            content = ' '.join(stream.read().upper().split())
        self.assertIn(
            'IF (N_VIRTUAL_FLAVOR_CONFIGS.GT.1) MC_HEL=0', content)

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

        self.assertEqual(content.count('CALLSREAL_STORE('), 8)
        self.assertEqual(
            content.count('RET_AMP_SPLIT,REAL_AMP_SPLIT'), 8)
        for routine in (
                'COMPUTE_SOFT_COUNTER_TERM',
                'COMPUTE_SOFT_COLLINEAR_COUNTER_TERM',
                'COMPUTE_COLLINEAR_COUNTER_TERM',
                'COMPUTE_REAL_EMISSION'):
            calls = re.findall(
                r'CALL%s\(([^)]*)\)' % routine, content)
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

        handle, path = tempfile.mkstemp(prefix='fks_info_', suffix='.inc')
        os.close(handle)
        try:
            exporter = export_fks.ProcessExporterFortranFKS.__new__(
                export_fks.ProcessExporterFortranFKS)
            writer = file_writers.FortranWriter(path)
            exporter.write_fks_info_file(
                writer, matrix_element, None)
            writer.close()
            with open(path) as stream:
                content = stream.read()
        finally:
            if os.path.exists(path):
                os.remove(path)

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


class IOExportFKSTest(IOTests.IOTestManager):
    """Test class for the export fks module"""


    def generate(self, process, model, multiparticles=[]):
        """Create a process"""

        def run_cmd(cmd):
            opt = dict(interface.options)
            opt['ninja'] = None
            opt['collier'] = None
            with misc.TMP_variable(interface, 'options', opt):
                interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        
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
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['p p > t t~ QED^2=0 QCD^2=4 [real=QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_ppw_fksall(self):
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['p p > w+ QED^2=2 QCD^2=0 [QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_tdecay_fksreal(self):
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['t > j j b QED^2=4 QCD^2=0 [real=QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_pptt_fks_loonly(self):
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['p p > t t~ QED^2=0 QCD^2=4 [LOonly=QCD]'], 'sm')

    @IOTests.createIOTest()
    def testIO_test_wprod_fksew(self):
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['p p > e+ ve QED^2=4 QCD^2=0 [QED]'], 'loop_qcd_qed_smCMS')

    @IOTests.createIOTest()
    def testIO_test_pptt_fksrealew(self):
        r""" target: SubProcesses/[P0.*\/.+\.(inc|f)]"""
        self.generate(['p p > t t~ QED^2=0 QCD^2=4 [real=QED]'], 'sm', 
                      multiparticles = ['p = u u~ d d~ s s~ c c~ g a'])



class TestFKSOutput(unittest.TestCase):
    """ this class is to test that the new and old nlo generation give
    identical results
    """

    def tearDown(self):
        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        run_cmd('set low_mem_multicore_nlo_generation False')
        run_cmd('set OLP MadLoop')

    def test_w_nlo_gen_qcd(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results for p p > e+ ve [QCD]
        """
        path = tempfile.mkdtemp('', 'TMPWTest', None)

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        
        run_cmd('set low_mem_multicore_nlo_generation True')
        run_cmd('generate p p > e+ ve QED^2=4 QCD^2=0 [QCD]')
        run_cmd('output %s' % os.path.join(path, 'W-newway'))
        run_cmd('set low_mem_multicore_nlo_generation False')
        run_cmd('generate p p > e+ ve QED^2=4 QCD^2=0 [QCD]')
        run_cmd('output %s' % os.path.join(path, 'W-oldway'))
        
        # the P0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', '*.f')) + \
           [os.path.join(path, 'W-oldway', 'SubProcesses', 'proc_characteristics')]):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)

        # the V0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', 'V0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', 'V0*', '*.f'))):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)


#    @test_manager.bypass_for_py3
    def test_w_nlo_gen_qed(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results for p p > e+ ve [QED]
        """
        path = tempfile.mkdtemp('', 'TMPWTest', None)

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        
        run_cmd('set low_mem_multicore_nlo_generation True')
        run_cmd('generate p p > e+ ve QED^2=4 QCD^2=0 [QED]')
        run_cmd('output %s' % os.path.join(path, 'W-newway'))
        run_cmd('set low_mem_multicore_nlo_generation False')
        run_cmd('generate p p > e+ ve QED^2=4 QCD^2=0 [QED]')
        run_cmd('output %s' % os.path.join(path, 'W-oldway'))
        
        # the P0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', '*.f')) + \
           [os.path.join(path, 'W-oldway', 'SubProcesses', 'proc_characteristics')]):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)

        # the V0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', 'V0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', 'V0*', '*.f'))):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)


 #   @test_manager.bypass_for_py3
    def test_z_nlo_gen_qed(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results for p p > e+ e- [QED], in particular
        that b-initiated processes are NOT combined with d-s ones
        """
        path = tempfile.mkdtemp('', 'TMPWTest', None)

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        
        run_cmd('define p3 = d s b d~ s~ b~ a')
        run_cmd('set low_mem_multicore_nlo_generation True')
        run_cmd('generate p3 p3 > e+ e- QED^2=4 QCD^2=0 [QED]')
        run_cmd('output %s' % os.path.join(path, 'Z-newway'))
        run_cmd('set low_mem_multicore_nlo_generation False')
        run_cmd('generate p3 p3 > e+ e- QED^2=4 QCD^2=0 [QED]')
        run_cmd('output %s' % os.path.join(path, 'Z-oldway'))
        
        # the P0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'Z-oldway', 'SubProcesses', 'P0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'Z-oldway', 'SubProcesses', 'P0*', '*.f')) + \
           [os.path.join(path, 'Z-oldway', 'SubProcesses', 'proc_characteristics')]):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)

        # the V0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'Z-oldway', 'SubProcesses', 'P0*', 'V0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'Z-oldway', 'SubProcesses', 'P0*', 'V0*', '*.f'))):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)

#    @test_manager.bypass_for_py3
    def test_z_nlo_gen_qcd(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results for p p > e+ e- [QED], in particular
        that the aa initiated folder is generated without virtuals
        """
        path = tempfile.mkdtemp('', 'TMPWTest', None)

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        
        run_cmd('define p3 = d s b d~ s~ b~')
        run_cmd('set low_mem_multicore_nlo_generation True')
        run_cmd('generate p3 p3 > e+ e- QED^2=4 QCD^2=0 [QCD]')
        run_cmd('output %s' % os.path.join(path, 'Z-newway'))
        run_cmd('set low_mem_multicore_nlo_generation False')
        run_cmd('generate p3 p3 > e+ e- QED^2=4 QCD^2=0 [QCD]')
        run_cmd('output %s' % os.path.join(path, 'Z-oldway'))
        
        # the P0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'Z-oldway', 'SubProcesses', 'P0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'Z-oldway', 'SubProcesses', 'P0*', '*.f')) + \
           [os.path.join(path, 'Z-oldway', 'SubProcesses', 'proc_characteristics')]):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)

        # the V0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'Z-oldway', 'SubProcesses', 'P0*', 'V0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'Z-oldway', 'SubProcesses', 'P0*', 'V0*', '*.f'))):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)

#    @test_manager.bypass_for_py3
    def test_wj_loonly_gen(self):
        """check that the new (memory and cpu efficient) and old generation
        mode at NLO give the same results for p p > w j [LOonly=QCD]
        """
        path = tempfile.mkdtemp('', 'TMPWTest', None)

        def run_cmd(cmd):
            interface.exec_cmd(cmd, errorhandling=False, printcmd=False, 
                               precmd=True, postcmd=True)

        interface = MGCmd.MasterCmd()
        

        run_cmd('set low_mem_multicore_nlo_generation True')
        run_cmd('generate p p > w+ j [LOonly=QCD]')
        run_cmd('output %s' % os.path.join(path, 'W-newway'))
        run_cmd('set low_mem_multicore_nlo_generation False')
        run_cmd('generate p p > w+ j [LOonly=QCD]')
        run_cmd('output %s' % os.path.join(path, 'W-oldway'))
                
        # the P0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', '*.f')) + \
           [os.path.join(path, 'W-oldway', 'SubProcesses', 'proc_characteristics')]):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)

        # the V0 dirs
        for oldf in \
          (glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', 'V0*', '*.inc')) + \
           glob.glob(os.path.join(path, 'W-oldway', 'SubProcesses', 'P0*', 'V0*', '*.f'))):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)

        # the Source dir 
        for oldf in \
          (glob.glob(os.path.join(path, 'W-oldway', 'Source', '*.inc')) + \
           glob.glob(os.path.join(path, 'W-oldway', 'Source', '*.f'))):
            
            if os.path.islink(oldf): 
                continue

            newf = oldf.replace('oldway', 'newway')

            for old_l, new_l in zip(open(oldf), open(newf)):
                self.assertEqual(old_l, new_l)


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
