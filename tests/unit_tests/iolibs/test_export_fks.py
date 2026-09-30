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
import madgraph.core.base_objects as base_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.export_fks as export_fks
import madgraph.iolibs.export_v4 as export_v4
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
