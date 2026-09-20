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
            content = open(path).read()
        finally:
            if os.path.exists(path):
                os.remove(path)

        self.assertIn('INTEGER REAL_FLAVOR_INDEX_D(16)', content)
        self.assertIn('INTEGER BORN_FLAVOR_INDEX_D(16)', content)
        self.assertIn('INTEGER VIRTUAL_FLAVOR_INDEX_D(16)', content)
        self.assertIn('PARAMETER (NBORN_FLAVOR_CONFIGS=4)', content)
        self.assertIn('data born_fks_config_d / 1, 2, 3, 4 /',
                      content.lower())
        self.assertIn('data real_flavor_index_d / 1, 2, 3, 4,',
                      content.lower())
        self.assertIn('data born_flavor_index_d / 1, 2, 3, 4,',
                      content.lower())
        self.assertNotIn(' 81,', content)
        self.assertNotIn('(-0.333', content)

        real_me = matrix_element.real_processes[0].matrix_element
        pdf_vars, pdf_data, pdf_lines, ee_vars = \
            exporter.get_pdf_lines_mir(real_me, 2)
        self.assertIn("INCLUDE 'fks_info.inc'", pdf_vars)
        self.assertIn('PDG_TYPE_D(NFKSPROCESS,1)', pdf_lines)
        self.assertIn('PDG_TYPE_D(NFKSPROCESS,2)', pdf_lines)
        self.assertIn('PD(IPROC)=FKS_PDF1*FKS_PDF2', pdf_lines)
        self.assertNotIn('_QUARK', pdf_vars + pdf_data + pdf_lines + ee_vars)

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


