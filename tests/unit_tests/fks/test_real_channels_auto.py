"""Default of the real-emission integration channels (real_channels_x0),
chosen at output from the real-emission diagrams."""

import os
import shutil
import tempfile
import unittest

import madgraph.interface.master_interface as mgcmd
import madgraph.fks.fks_helas_objects as fks_helas
import madgraph.fks.fks_real_channels_auto as auto
from madgraph.various import banner


class TestRealChannelsDecision(unittest.TestCase):
    """The diagram analysis on generated processes"""

    def decide(self, model, process):
        interface = mgcmd.MasterCmd()
        interface.no_notification()
        interface.exec_cmd('import model %s' % model, printcmd=False)
        interface.exec_cmd('generate %s' % process, printcmd=False)
        mes = fks_helas.FKSHelasMultiProcess(interface._fks_multi_proc)
        directories = [auto.real_structures(me)
                       for me in mes['matrix_elements']]
        return auto.decide(directories, 10)

    def test_photon_initiated_t_channel(self):
        """u a > w+ h d: the photon couples to the t-channel W, which no
        Born diagram (s-channel W only) has"""
        x0, summary = self.decide('loop_qcd_qed_sm', 'p p > w+ h [real=QED]')
        self.assertEqual(x0, auto.AUTO_X0)
        uncovered = dict((tuple(pdgs), diags) for pdgs, diags, n
                         in summary['reals_uncovered'])
        self.assertIn((2, 22, 1, 24, 25), uncovered)
        # q q~ > w+ h a is a pure emission off the Born
        self.assertNotIn((2, -1, 22, 24, 25), uncovered)

    def test_gluon_fusion_in_real_emission(self):
        """u g > w+ b b~ d: g g > b b~ through a t-channel gluon off the
        light-quark line (4FS)"""
        x0, summary = self.decide('loop_sm', 'p p > w+ b b~ [real=QCD]')
        self.assertEqual(x0, auto.AUTO_X0)
        self.assertTrue(all(21 in pdgs[:2]
                            for pdgs, diags, n in summary['reals_uncovered']))

    def test_born_like_real_emission(self):
        """Drell-Yan and t t~ at NLO QCD: all the real-emission diagrams
        are followed by the Born channels"""
        for process in ['p p > e+ e- [real=QCD]', 'p p > t t~ [real=QCD]']:
            x0, summary = self.decide('loop_sm', process)
            self.assertEqual(x0, -1., process)
            self.assertEqual(summary['n_uncovered'], 0, process)

    def test_too_many_channels(self):
        x0, summary = auto.decide([], 0)
        self.assertEqual(x0, -1.)
        x0, summary = self.decide('loop_qcd_qed_sm', 'p p > w+ h [real=QED]')
        self.assertEqual(x0, auto.AUTO_X0)
        directories = [{'born': {'legs': [], 'props': []}, 'reals': []}]
        self.assertEqual(auto.decide(directories,
                                     auto.MAX_HARD_CHANNELS + 1)[0], -1.)


class TestRealChannelsSectors(unittest.TestCase):
    """The coverage rule on hand-made sectors"""

    def test_identical_permutations(self):
        legs = [(1, 21, False), (2, 21, False), (3, 21, True),
                (4, 21, True), (5, 21, True)]
        sectors = auto.identical_permutations([(5, 4, True, 4)], legs)
        self.assertIn((3, 4, True, 4), sectors)
        self.assertIn((5, 3, True, 4), sectors)
        self.assertEqual(len(sectors), 6)

    def test_follows(self):
        # 1 2 > 3 4 5, i=5 soft, j=4 (Born leg 4), Born 1 2 > 3 4
        mapping = {1: 1, 2: 2, 3: 3, 4: 4}
        born_props = set([(frozenset([2, 4]), 6)])
        sector = auto.Sector(5, 4, True, mapping, born_props, 4)
        # does not separate i from j
        self.assertTrue(sector.follows({4, 5}, 21, 5))
        self.assertTrue(sector.follows({1, 3}, 21, 5))
        # i attached to the external leg 3: soft enhancement only
        self.assertTrue(sector.follows({3, 5}, 6, 5))
        # i emitted off a propagator that the Born has
        self.assertTrue(sector.follows({2, 4}, 6, 5))
        # a propagator that the Born does not have
        self.assertFalse(sector.follows({2, 4}, 24, 5))
        hard = auto.Sector(5, 4, False, mapping, born_props, 4)
        self.assertFalse(hard.follows({3, 5}, 6, 5))


class TestRealChannelsDefault(unittest.TestCase):
    """The default of real_channels_x0 written in the run_card of a process"""

    def make_card(self, x0):
        charac = banner.ProcCharacteristic()
        charac['real_channels_x0'] = x0
        charac['ninitial'] = 2
        process = self.process
        card = banner.RunCardNLO()
        card.create_default_for_process(charac, None, [process])
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        path = os.path.join(directory, 'run_card.dat')
        card.write(path)
        return card, open(path).read(), path

    def setUp(self):
        interface = mgcmd.MasterCmd()
        interface.no_notification()
        interface.exec_cmd('import model loop_sm', printcmd=False)
        interface.exec_cmd('generate u u~ > e+ e- [real=QCD]', printcmd=False)
        self.process = \
            interface._fks_multi_proc['born_processes'][0].born_amp['process']

    def test_written_in_run_card(self):
        for x0 in (0.05, -1.0):
            card, text, path = self.make_card(x0)
            self.assertEqual(card['real_channels_x0'], x0)
            line = next(l for l in text.splitlines()
                        if '= real_channels_x0' in l)
            self.assertEqual(float(line.split('=')[0]), x0)
            self.assertEqual(banner.RunCardNLO(path)['real_channels_x0'], x0)
