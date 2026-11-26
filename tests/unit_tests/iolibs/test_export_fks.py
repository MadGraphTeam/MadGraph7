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
import json
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
import madgraph.fks.fks_helas_objects as fks_helas
import madgraph.core.base_objects as base_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.export_fks as export_fks
import madgraph.iolibs.export_v4 as export_v4
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
        # Helicity -1 has the same full matrix element as +1 and is not
        # recomputed; helicity 0 has none (its two diagrams cancel) and is
        # skipped. Both have nonzero diagram squares, which the weights
        # leave out at every point, the first one included.
        source = template % {
            'proc_prefix': '1', 'info_lines': '', 'process_lines': '',
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
      double precision p(0:3,1),ans(0:1)
      include 'real_matrix_weights.inc'
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
                'matrix.f': source,
                'nexternal.inc': '      integer nexternal\n      parameter(nexternal=1)\n',
                'coupl.inc': '',
                'real_matrix_weights.inc': (
                    '      double precision real_amp2(2)\n'
                    '      common /c_real_diagram_weights/ real_amp2\n'
                    '      logical get_real_diagram_weights\n'
                    '      common /c_get_real_diagram_weights/ get_real_diagram_weights\n')
            }.items():
                with open(os.path.join(path, name), 'w') as stream:
                    stream.write(content)
            subprocess.check_output([compiler, '-ffixed-line-length-none',
                                     '-fcheck=all', aloha_functions,
                                     'matrix.f', '-o', 'check'],
                                    cwd=path, stderr=subprocess.STDOUT)
            subprocess.check_output([os.path.join(path, 'check')], cwd=path,
                                    stderr=subprocess.STDOUT)


class TestFKSConfigurationsJson(unittest.TestCase):
    """fks_configurations.json, the FKS configurations of fks_info.inc for
    MadSpace"""

    @staticmethod
    def fksborn(info_list, real_processes=()):
        legs = [{'id': 2, 'state': False}, {'id': 21, 'state': False},
                {'id': 2, 'state': True}, {'id': 11, 'state': True},
                {'id': -11, 'state': True}]
        return SimpleNamespace(born_me={'processes': [{'legs': legs}]},
                               get_fks_info_list=lambda: info_list,
                               real_processes=list(real_processes))

    @staticmethod
    def real(colors, charges, fks_j_from_i, tags=None, flavors=()):
        tags = tags if tags is not None else [False] * len(colors)
        processes = [{'legs': [{'id': pdg} for pdg in pdgs]} for pdgs in flavors]
        return SimpleNamespace(colors=colors, charges=charges,
                               particle_tags=tags,
                               fks_j_from_i=fks_j_from_i, isfinite=False,
                               matrix_element={'processes': processes})

    @staticmethod
    def info(n_me, pdgs, i, j, ij_id, soft):
        return {'n_me': n_me, 'pdgs': pdgs,
                'fks_info': {'i': i, 'j': j, 'ij': j, 'ij_id': ij_id,
                             'need_color_links': soft,
                             'need_charge_links': False,
                             'splitting_type': ['QCD'],
                             'extra_cnt_index': -1,
                             'underlying_born': [[2, 21, 2, 11, -11]],
                             'rb_links': []}}

    def test_configurations(self):
        # u g > u e- e+ g, i the gluon off the incoming u, and g g > u e- e+
        # u~, i the u~ off the incoming gluon (which the Born has as a u)
        charges = [2 / 3, 0., 2 / 3, -1., 1.]
        fksborn = self.fksborn([
            self.info(1, [2, 21, 2, 11, -11, 21], 6, 1, 2, True),
            self.info(2, [21, 21, 2, 11, -11, -2], 6, 1, 2, False)], [
            self.real([3, 8, 3, 1, 1, 8], charges + [0.],
                      {1: [], 2: [], 3: [], 4: [], 5: [], 6: [1, 2, 3]},
                      [False, False, False, True, False, False],
                      [[2, 21, 2, 11, -11, 21], [4, 21, 4, 11, -11, 21]]),
            self.real([8, 8, 3, 1, 1, -3], [0., 0.] + charges[2:] + [-2 / 3],
                      {1: [], 2: [], 3: [], 4: [], 5: [], 6: [2]},
                      flavors=[[21, 21, 2, 11, -11, -2],
                               [21, 21, 4, 11, -11, -4]])])
        data = export_fks.get_fks_configurations(fksborn)
        self.assertEqual(data['nincoming'], 2)
        self.assertEqual(data['born_pdgs'], [2, 21, 2, 11, -11])
        self.assertEqual(data['born_flavors'], [[2, 21, 2, 11, -11]])
        self.assertEqual(data['real_flavors'], [
            [[2, 21, 2, 11, -11, 21], [4, 21, 4, 11, -11, 21]],
            [[21, 21, 2, 11, -11, -2], [21, 21, 4, 11, -11, -4]]])
        self.assertEqual(data['configurations'], [
            {'i_fks': 6, 'j_fks': 1, 'ij_pdg': 2,
             'real_pdgs': [2, 21, 2, 11, -11, 21], 'real_me': 1,
             'need_color_links': True, 'need_charge_links': False,
             'splitting_type': ['QCD'], 'extra_cnt_index': -1,
             'real_colors': [3, 8, 3, 1, 1, 8],
             'real_charges': charges + [0.],
             'real_tags': [False, False, False, True, False, False],
             'fks_j_from_i': [[], [], [], [], [], [1, 2, 3]]},
            {'i_fks': 6, 'j_fks': 1, 'ij_pdg': 2,
             'real_pdgs': [21, 21, 2, 11, -11, -2], 'real_me': 2,
             'need_color_links': False, 'need_charge_links': False,
             'splitting_type': ['QCD'], 'extra_cnt_index': -1,
             'real_colors': [8, 8, 3, 1, 1, -3],
             'real_charges': [0., 0.] + charges[2:] + [-2 / 3],
             'real_tags': [False] * 6,
             'fks_j_from_i': [[], [], [], [], [], [2]]}])
        exporter = object.__new__(export_fks.ProcessExporterFortranFKS)
        with tempfile.TemporaryDirectory() as path:
            filename = os.path.join(path, 'fks_configurations.json')
            exporter.write_fks_configurations_json(filename, fksborn)
            with open(filename) as stream:
                self.assertEqual(json.load(stream), data)

    def test_no_configurations(self):
        # [LOonly]: none, not the fake configuration of fks_info.inc
        data = export_fks.get_fks_configurations(self.fksborn([]))
        self.assertEqual(data['configurations'], [])
        self.assertEqual(data['real_flavors'], [])
        self.assertEqual(data['extra_counterterms'], [])
        self.assertEqual(data['born_pdgs'], [2, 21, 2, 11, -11])

    def test_extra_counterterms(self):
        # e+ e- > u u~ g at alpha_s alpha^2 with alpha^3 allowed: the d d~ of
        # e+ e- > u u~ d d~ comes from the gluon or from a photon, whose Born
        # e+ e- > u u~ a is the extra counterterm
        legs = [{'id': pdg, 'state': k >= 2}
                for k, pdg in enumerate([-11, 11, 2, -2, 22])]
        extra = type('ExtraCnt', (), {
            '__getitem__': lambda self, key: {'processes': [{'legs': legs}]}[key],
            'get_split_orders_mapping':
                lambda self: ([(0, 6)], [((0, 3), (1, 2))])})()
        fksborn = self.fksborn([])
        fksborn.extra_cnt_me_list = [extra]
        orders = {'split_orders': ['QCD', 'QED'],
                  'amp_split_orders': [(2, 4), (2, 6), (4, 4), (0, 8)],
                  'amp_split_size_born': 1,
                  'max_born_orders': {'QCD': 2, 'QED': 6},
                  'max_nlo_orders': {'QCD': 4, 'QED': 8}}
        data = export_fks.get_fks_configurations(fksborn, orders)
        self.assertEqual(data['extra_counterterms'],
                         [{'born_pdgs': [-11, 11, 2, -2, 22]}])
        self.assertEqual(data['orders'], {
            'split_orders': ['QCD', 'QED'], 'born': [[2, 4]],
            'nlo': [[2, 6], [4, 4], [0, 8]], 'max_born': [2, 6],
            'max_nlo': [4, 8], 'extra_born': [[[0, 6]]]})

    @staticmethod
    def sudakov_born(pdgs, squared_orders, born_sq_orders, model, helicities):
        """A Born matrix element of --ewsudakov: its process, split orders and
        helicities."""
        legs = [{'id': pdg, 'state': k >= 2} for k, pdg in enumerate(pdgs)]
        process = {'legs': legs, 'split_orders': ['QCD', 'QED'],
                   'born_sq_orders': born_sq_orders,
                   'model': {'name': model}}
        return type('SudakovBorn', (), {
            '__getitem__': lambda self, key: {'processes': [process]}[key],
            'get_split_orders_mapping': lambda self: (squared_orders, []),
            'get_helicity_matrix': lambda self: helicities})()

    def test_ewsudakov(self):
        # u u~ > w+ w- with the Goldstone amplitude of both W bosons and one
        # interference of each kind
        helicities = [[1, -1, 0, 0], [-1, 1, 1, -1]]
        fksborn = self.fksborn([])
        fksborn.born_me = self.sudakov_born(
            [2, -2, 24, -24], [(0, 4)], {'QCD': 0, 'QED': 4},
            'loop_qcd_qed_sm_Gmu_forSudakov', helicities)

        def me(pdgs):
            legs = [{'id': pdg} for pdg in pdgs]
            return {'matrix_element': {'processes': [{'legs': legs}]}}

        fksborn.sudakov_matrix_elements = [
            dict(me([2, -2, 251, -251]), type='goldstone', base_amp=0,
                 legs=[{'number': 3}, {'number': 4}],
                 pdgs=[[24, -24], [251, -251]]),
            dict(me([1, -1, 24, -24]), type='ipm2', base_amp=0,
                 legs=[{'number': 1}, {'number': 2}], pdgs=[[2, -2], [1, -1]]),
            # as write_sudakov_me leaves them, padded with zeros
            dict(me([2, -2, 25, -251]), type='iz1', base_amp=1,
                 legs=[{'number': 3}], pdgs=[[251, 0], [25, 0]]),
        ]
        fksborn.ewsudakov = True
        data = export_fks.get_fks_configurations(fksborn)
        self.assertEqual(data['ewsudakov'], {
            'scheme': 'gmu', 'has_lo1': True, 'has_lo2': False,
            'lo1': [0, 4], 'lo2': None, 'helicities': helicities,
            'goldstone': [{'legs': [3, 4], 'pdgs': [2, -2, 251, -251]}],
            'interference': [
                {'type': 'ipm2', 'base': 0, 'legs': [1, 2],
                 'old_pdgs': [2, -2], 'new_pdgs': [1, -1],
                 'pdgs': [1, -1, 24, -24]},
                {'type': 'iz1', 'base': 1, 'legs': [3],
                 'old_pdgs': [251], 'new_pdgs': [25],
                 'pdgs': [2, -2, 25, -251]}]})
        # a LO2, alpha(MZ) scheme
        fksborn.born_me = self.sudakov_born(
            [2, -2, 6, -6], [(4, 0), (2, 2), (0, 4)], {'QCD': 4, 'QED': 0},
            'loop_qcd_qed_sm_forSudakov', helicities)
        fksborn.sudakov_matrix_elements = []
        data = export_fks.get_fks_configurations(fksborn)['ewsudakov']
        self.assertEqual(
            [data[k] for k in ['scheme', 'has_lo1', 'has_lo2', 'lo1', 'lo2']],
            ['alpha_mz', True, True, [4, 0], [2, 2]])
        self.assertEqual(export_fks.get_ewsudakov_lo(fksborn)['lo2_pos'], 2)
        # more than one coupling combination within the orders of the Born
        fksborn.born_me = self.sudakov_born(
            [2, -2, 6, -6], [(4, 0), (2, 2)], {'QCD': 4, 'QED': 2},
            'loop_qcd_qed_sm_forSudakov', helicities)
        self.assertRaises(export_fks.MadGraph5Error,
                          export_fks.get_fks_configurations, fksborn)


class TestBornGscale(unittest.TestCase):
    """gscale: when only alpha_S changed, born.f rescales its cached Born
    amplitudes by (G'/G)**n instead of recomputing them (BORN_GSCALE). With
    several split orders each one has its own power n, which must follow the
    QCD order of the amplitudes, and each coupling is checked against the
    QCD order of its interaction."""

    @classmethod
    def setUpClass(cls):
        interface = MGCmd.MasterCmd()
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


class TestSplitOrdersInfo(unittest.TestCase):
    """The coupling orders of orders.inc and amp_split_orders.inc, which
    fks_configurations.json also carries for MadSpace."""

    @classmethod
    def setUpClass(cls):
        interface = MGCmd.MasterCmd()
        interface.do_import('model loop_sm')
        # three Born orders (QCD^2 = 4, 2, 0), each with its QCD corrections
        interface.do_generate('u u~ > t t~ QCD^2<=4 QED^2<=4 [real=QCD]')
        helas = fks_helas.FKSHelasMultiProcess(interface._fks_multi_proc)
        cls.matrix_element = helas.get_matrix_elements()[0]

    def test_orders(self):
        info = export_fks.get_split_orders_info(self.matrix_element)
        self.assertEqual(info['split_orders'], ['QCD', 'QED'])
        self.assertEqual((info['qcd_pos'], info['qed_pos']), (1, 2))
        born_count = info['amp_split_size_born']
        born = info['amp_split_orders'][:born_count]
        nlo = info['amp_split_orders'][born_count:]
        self.assertEqual(sorted(born), [[0, 4], [2, 2], [4, 0]])
        # every NLO combination is one of a Born with two more powers of QCD
        # or QED (the latter from reals with only an interference)
        for orders in nlo:
            self.assertTrue([orders[0] - 2, orders[1]] in born
                            or [orders[0], orders[1] - 2] in born, orders)
        self.assertEqual(sorted(nlo), [[0, 6], [2, 4], [4, 2], [6, 0]])
        data = export_fks.get_fks_configurations(self.matrix_element, info)
        self.assertEqual(data['orders'], {
            'split_orders': ['QCD', 'QED'], 'born': born, 'nlo': nlo,
            'max_born': [info['max_born_orders'][o] for o in ['QCD', 'QED']],
            'max_nlo': [info['max_nlo_orders'][o] for o in ['QCD', 'QED']],
            'extra_born': []})

    def test_same_as_orders_inc(self):
        """write_orders_file writes the same combinations."""
        exporter = object.__new__(export_fks.ProcessExporterFortranFKS)
        with tempfile.TemporaryDirectory() as path:
            filename = os.path.join(path, 'orders.inc')
            amp_split_orders, size, size_born = exporter.write_orders_file(
                writers.FortranWriter90(filename), self.matrix_element)
        info = export_fks.get_split_orders_info(self.matrix_element)
        self.assertEqual([list(o) for o in amp_split_orders],
                         info['amp_split_orders'])
        self.assertEqual(size_born, info['amp_split_size_born'])


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
    ignored_files = ['amcatnlo.tar.gz', 'index.html',
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
