################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
################################################################################
"""MadSpin in an mg7 gridpack: the card a gridpack runs, the seed of its
MadSpin step, the shadow ms_dir every run works in, the Python it ships,
[gridpack] include_madspin, and the matrix-element cache of a reused ms_dir.
The end-to-end runs are in tests/acceptance_tests/test_mg7_gridpack_madspin.py.
"""

import importlib.util
import os
import pickle
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from MadSpin import gridpack_decay
from madgraph.mg7 import gridpack_madspin
from madgraph.various.banner import RunCardMG7

pjoin = os.path.join
_ROOT = Path(__file__).resolve().parents[3]

_CARD = """\
# a comment
set seed 77
set ms_dir /somewhere/ms
import /some/events.lhe.gz
set spinmode madspin
define l+ = e+ mu+
decay t > w+ b, w+ > l+ vl
decay t~ > w- b~, w- > l- vl~
launch
"""


class TestGridpackCard(unittest.TestCase):

    def test_run_lines_are_dropped(self):
        body = gridpack_decay.gridpack_card(_CARD)
        for line in ('set seed', 'set ms_dir', 'import ', 'launch'):
            self.assertNotIn(line, body)
        for line in ('# a comment', 'set spinmode madspin', 'define l+ = e+ mu+',
                     'decay t > w+ b, w+ > l+ vl', 'decay t~ > w- b~, w- > l- vl~'):
            self.assertIn(line, body)

    def test_refused(self):
        for line in ('import model sm-full /tmp/card.dat',
                     'set spinmode madspin_v1', 'set spinmode onshell_v1',
                     'set decay_generator madevent'):
            with self.assertRaises(gridpack_decay.GridpackDecayError, msg=line):
                gridpack_decay.gridpack_card(line + '\ndecay t > w+ b\n')
        # the modes it supports pass
        for mode in ('madspin', 'full', 'onshell', 'none', 'PA'):
            gridpack_decay.gridpack_card('set spinmode %s\n' % mode)

    def test_command_file(self):
        body = gridpack_decay.gridpack_card(_CARD)
        text = gridpack_decay.run_card(body, 'events.lhe', 'ms_dir', 12, nb_core=3)
        lines = text.splitlines()
        self.assertEqual(lines[:3], ['import events.lhe', 'set ms_dir ms_dir',
                                     'set seed 12'])
        # the caller's cores after the card, then the launch
        self.assertEqual(lines[-2:], ['set nb_core 3', 'launch'])
        prepare = gridpack_decay.run_card(body, 'events.lhe', 'ms_dir', None,
                                          prepare=True)
        self.assertNotIn('set seed', prepare)
        self.assertTrue(prepare.endswith('launch --prepare_only\n'))


def _load_runner():
    """template_files/mg7/gridpack.py, with what it imports at load time
    replaced: it runs without MadGraph, and madspace may not be built."""
    path = _ROOT / 'madgraph' / 'iolibs' / 'template_files' / 'mg7' / 'gridpack.py'
    setup = types.ModuleType('gridpack_setup')
    setup.__getattr__ = lambda name: None
    spec = importlib.util.spec_from_file_location('mg7_gridpack_runner', path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {'madspace': types.ModuleType('madspace'),
                                       'gridpack_setup': setup}):
        spec.loader.exec_module(module)
    return module


class TestMadSpinSeed(unittest.TestCase):

    def test_range_and_reproducibility(self):
        seeds = [0, 1, 42, 2 ** 63 - 1, 30081 ** 2, -5]
        for seed in seeds:
            value = gridpack_decay.madspin_seed(seed)
            self.assertGreaterEqual(value, 1)
            self.assertLessEqual(value, 30081 ** 2)
            self.assertEqual(value, gridpack_decay.madspin_seed(seed))
        self.assertEqual(len({gridpack_decay.madspin_seed(s) for s in range(200)}),
                         200)

    def test_documented_mapping(self):
        import hashlib
        seed = 123456789012345
        expected = int.from_bytes(
            hashlib.sha256(b'123456789012345:madspin').digest(), 'big') \
            % (30081 ** 2 - 1) + 1
        self.assertEqual(gridpack_decay.madspin_seed(seed), expected)

    def test_gridpack_runner_agrees(self):
        """bin/generate_events computes it without importing MadGraph."""
        runner = _load_runner()
        for seed in (1, 7, 2 ** 62 + 3):
            self.assertEqual(runner.madspin_seed(seed),
                             gridpack_decay.madspin_seed(seed))


class TestShadowMsDir(unittest.TestCase):

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='ms_shadow_')
        self.source = pjoin(self.path, 'ms_dir')
        os.makedirs(pjoin(self.source, 'decay_6_0', 'Events'))
        os.makedirs(pjoin(self.source, 'decay_6_0', 'data'))
        os.makedirs(pjoin(self.source, 'decay_6_0', 'bin'))
        with open(pjoin(self.source, 'decay_6_0', 'ms_partial_width.dat'), 'w') as f:
            f.write('0.3\n')
        os.makedirs(pjoin(self.source, 'madspin_me', 'Cards'))
        os.makedirs(pjoin(self.source, 'madspin_me', 'SubProcesses'))
        with open(pjoin(self.source, 'madspin_me', 'Cards', 'param_card.dat'), 'w') as f:
            f.write('card\n')
        os.makedirs(pjoin(self.source, 'other'))
        for name in ('max_wgt', 'madspin_card.dat', 'param_card.dat'):
            with open(pjoin(self.source, name), 'w') as f:
                f.write(name)

    def tearDown(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def test_layout(self):
        target = gridpack_decay.shadow_ms_dir(self.source, pjoin(self.path, 'run'))
        # the small files of the top are copies: a run rewrites some of them
        for name in ('max_wgt', 'madspin_card.dat', 'param_card.dat'):
            self.assertFalse(os.path.islink(pjoin(target, name)))
        self.assertTrue(os.path.islink(pjoin(target, 'other')))
        decay = pjoin(target, 'decay_6_0')
        self.assertFalse(os.path.islink(decay))
        self.assertTrue(os.path.islink(pjoin(decay, 'data')))
        self.assertTrue(os.path.islink(pjoin(decay, 'bin')))
        # this run's pools
        self.assertFalse(os.path.islink(pjoin(decay, 'Events')))
        self.assertFalse(os.path.islink(pjoin(decay, 'ms_partial_width.dat')))
        me = pjoin(target, 'madspin_me')
        self.assertTrue(os.path.islink(pjoin(me, 'SubProcesses')))
        self.assertFalse(os.path.islink(pjoin(me, 'Cards')))

    def test_writes_stay_in_the_shadow(self):
        target = gridpack_decay.shadow_ms_dir(self.source, pjoin(self.path, 'run'))
        os.makedirs(pjoin(target, 'decay_6_0', 'Events', 'run_01'))
        with open(pjoin(target, 'madspin_me', 'Cards', 'param_card.dat'), 'w') as f:
            f.write('other card\n')
        with open(pjoin(target, 'param_card.dat'), 'w') as f:
            f.write('changed')
        self.assertEqual(os.listdir(pjoin(self.source, 'decay_6_0', 'Events')), [])
        with open(pjoin(self.source, 'madspin_me', 'Cards', 'param_card.dat')) as f:
            self.assertEqual(f.read(), 'card\n')
        with open(pjoin(self.source, 'param_card.dat')) as f:
            self.assertEqual(f.read(), 'param_card.dat')

    def test_clean_prepared(self):
        os.makedirs(pjoin(self.source, 'decay_6_0', 'Events', 'run_01'))
        for name in ('ms_wstatus_0', 'ms_gen_6.json', 'seeds.dat'):
            with open(pjoin(self.source, name), 'w') as f:
                f.write('x')
        gridpack_decay.clean_prepared(self.source)
        self.assertEqual(os.listdir(pjoin(self.source, 'decay_6_0', 'Events')), [])
        for name in ('ms_wstatus_0', 'ms_gen_6.json', 'seeds.dat'):
            self.assertFalse(os.path.exists(pjoin(self.source, name)), name)
        for name in ('max_wgt', 'madspin_card.dat', 'param_card.dat'):
            self.assertTrue(os.path.exists(pjoin(self.source, name)), name)

    def test_decay_needs_a_prepared_ms_dir(self):
        os.remove(pjoin(self.source, 'madspin_card.dat'))
        with self.assertRaisesRegex(gridpack_decay.GridpackDecayError,
                                    'not a prepared ms_dir'):
            gridpack_decay.decay(self.source, pjoin(self.path, 'events.lhe'), 1)


class TestInstallRuntime(unittest.TestCase):

    def test_copies_what_madspin_imports(self):
        path = tempfile.mkdtemp(prefix='ms_runtime_')
        try:
            root = pjoin(path, 'mg')
            for package in gridpack_decay.RUNTIME_PACKAGES:
                os.makedirs(pjoin(root, package, '__pycache__'))
                with open(pjoin(root, package, '__init__.py'), 'w') as f:
                    f.write('')
                with open(pjoin(root, package, '__pycache__', 'x.pyc'), 'w') as f:
                    f.write('')
            os.makedirs(pjoin(root, 'tests'))
            for name in ('VERSION', pjoin('vendor', '__init__.py'),
                         pjoin('models', '__init__.py'),
                         pjoin('Template', 'LO', 'Source', '.make_opts'),
                         pjoin('Template', 'Common', 'Source', 'PDF',
                               'lep_densities', 'isrbetmll', 'info'),
                         pjoin('Template', 'Common', 'Source', 'PDF',
                               'lep_densities', 'isrbetmll', 'eepdf.f'),
                         pjoin('input', 'mg7_configuration.txt')):
                os.makedirs(os.path.dirname(pjoin(root, name)), exist_ok=True)
                with open(pjoin(root, name), 'w') as f:
                    f.write(name)
            model = pjoin(path, 'models_elsewhere', 'my_model')
            os.makedirs(model)
            for name in ('__init__.py', 'particles.py', 'py3_model.pkl'):
                with open(pjoin(model, name), 'w') as f:
                    f.write('')
            target = gridpack_decay.install_runtime(pjoin(path, 'out'), model, root)
            for package in gridpack_decay.RUNTIME_PACKAGES:
                self.assertTrue(os.path.exists(pjoin(target, package, '__init__.py')))
                self.assertFalse(os.path.exists(pjoin(target, package, '__pycache__')))
            self.assertFalse(os.path.exists(pjoin(target, 'tests')))
            lep = pjoin(target, 'Template', 'Common', 'Source', 'PDF',
                        'lep_densities', 'isrbetmll')
            self.assertEqual(os.listdir(lep), ['info'])
            source = pjoin(target, 'Template', 'LO', 'Source')
            self.assertTrue(os.path.exists(pjoin(source, 'make_opts')))
            self.assertTrue(os.path.exists(pjoin(target, 'input',
                                                 'mg7_configuration.txt')))
            self.assertTrue(os.path.exists(pjoin(target, 'models', 'my_model',
                                                 'particles.py')))
            self.assertFalse(os.path.exists(pjoin(target, 'models', 'my_model',
                                                  'py3_model.pkl')))
        finally:
            shutil.rmtree(path, ignore_errors=True)


class TestIncludeMadSpin(unittest.TestCase):

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='include_madspin_')
        os.makedirs(pjoin(self.path, 'Cards'))
        self.cards = pjoin(self.path, 'Cards')

    def tearDown(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def card(self, value=None):
        card = RunCardMG7()
        if value is not None:
            card.set('gridpack.include_madspin', value, user=True)
        return card

    def test_default_is_auto(self):
        self.assertEqual(self.card()['gridpack']['include_madspin'], 'auto')

    def test_auto_follows_the_switch_and_the_card(self):
        wanted = gridpack_madspin.madspin_wanted
        self.assertFalse(wanted(self.card(), None, self.cards))
        self.assertFalse(wanted(self.card(), True, self.cards))
        with open(pjoin(self.cards, 'madspin_card.dat'), 'w') as f:
            f.write('decay t > w+ b\n')
        self.assertTrue(wanted(self.card(), None, self.cards))
        self.assertTrue(wanted(self.card(), True, self.cards))
        self.assertFalse(wanted(self.card(), False, self.cards))

    def test_explicit(self):
        wanted = gridpack_madspin.madspin_wanted
        with self.assertRaisesRegex(gridpack_madspin.GridpackMadSpinError,
                                    'no Cards/madspin_card.dat'):
            wanted(self.card(True), None, self.cards)
        with open(pjoin(self.cards, 'madspin_card.dat'), 'w') as f:
            f.write('decay t > w+ b\n')
        self.assertTrue(wanted(self.card(True), False, self.cards))
        self.assertFalse(wanted(self.card(False), True, self.cards))


class TestDensityMatrixElementCache(unittest.TestCase):
    """decay_all_events_onshell keeps the matrix elements of a persistent
    directory (ms_dir, use_old_dir): a cache hit generates and compiles
    nothing."""

    def setUp(self):
        import MadSpin.decay as madspin
        self.madspin = madspin
        self.path = tempfile.mkdtemp(prefix='ms_me_cache_')

    def tearDown(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def stub(self, persistent=True):
        stub = self.madspin.decay_all_events_onshell.__new__(
            self.madspin.decay_all_events_onshell)
        stub.path_me = self.path
        stub.options = {'ms_dir': self.path if persistent else '',
                        'use_old_dir': False}
        stub.all_me = {((21, 21), (-6, 6)): {'pdir': 'P1_gg_ttx',
                                            'order': ((21, 21), (6, -6)),
                                            'type': 'production'}}
        return stub

    def test_round_trip_and_key(self):
        stub = self.stub()
        stub.save_me_cache('key-1')
        self.assertTrue(os.path.exists(pjoin(self.path, stub.ME_CACHE)))
        other = self.stub()
        other.all_me = None
        self.assertEqual(other.load_me_cache('key-1'), stub.all_me)
        # other generate lines: generated again
        self.assertIsNone(other.load_me_cache('key-2'))

    def test_not_persistent(self):
        stub = self.stub(persistent=False)
        stub.save_me_cache('key')
        self.assertIsNone(stub.load_me_cache('key'))

    def test_unreadable(self):
        with open(pjoin(self.path, self.madspin.decay_all_events_onshell.ME_CACHE),
                  'wb') as f:
            f.write(b'not a pickle')
        self.assertIsNone(self.stub().load_me_cache('key'))

    def test_cache_hit_compiles_nothing(self):
        stub = self.stub()
        stub._me_from_cache = True
        stub.refresh_me_param_cards = lambda: None
        stub.compile_me_trees = mock.Mock(side_effect=AssertionError('compiled'))
        stub.compile()
        stub._me_from_cache = False
        stub._me_cache_pending = 'key-3'
        stub.compile_me_trees = mock.Mock()
        stub.compile()
        stub.compile_me_trees.assert_called_once_with()
        self.assertEqual(self.stub().load_me_cache('key-3'), stub.all_me)

    def test_density_modes_pickle_nothing(self):
        self.assertFalse(self.madspin.decay_all_events_onshell._pickle_status)
        self.assertFalse(self.madspin.decay_all_events_density._pickle_status)
        self.assertTrue(self.madspin.decay_all_events._pickle_status)


if __name__ == '__main__':
    unittest.main()
