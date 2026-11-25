################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""The precision check of an mg7 run (madgraph/mg7/precision_check.py), and
denom64 as the default precision.

What needs a compiler and madspace -- the two builds, the ctypes evaluation,
that it reproduces the |M|^2 of the run itself -- is in
tests/acceptance_tests/test_mg7_precision_check.py.
"""

from __future__ import absolute_import

import math
import os
import shutil
import tempfile
import unittest

# no madgraph.mg7.common here: it imports madspace, which the unit-test
# runners do not have when they collect the test modules
import madgraph.mg7.precision_check as precision_check
from madgraph.various.banner import RunCardMG7

pjoin = os.path.join


class TestPrecisionDefault(unittest.TestCase):
    """denom64 (FPTYPE=v) is the default precision everywhere."""

    def test_run_card_default_is_denom64(self):
        card = RunCardMG7()
        self.assertEqual(card['run']['precision'], 'denom64')
        self.assertEqual(card['run']['precision_check'], 0.0)
        self.assertEqual(card['run']['precision_check_max_events'], 1000)
        self.assertEqual(card['run']['precision_check_threshold'], 1e-3)

    def test_written_run_card_has_the_precision_check(self):
        tmpdir = tempfile.mkdtemp()
        try:
            path = pjoin(tmpdir, 'run_card.toml')
            RunCardMG7().write(path)
            with open(path) as f:
                text = f.read()
            self.assertIn('precision = "denom64"', text)
            self.assertIn('precision_check = 0.0', text)
            card = RunCardMG7(path)
            self.assertEqual(card['run']['precision_check'], 0.0)
        finally:
            shutil.rmtree(tmpdir)

    def test_makefile_default_is_v(self):
        """madmatrix.mk builds FPTYPE=v without an FPTYPE, in a build
        directory of its own per FPTYPE."""
        import madgraph
        mk = pjoin(os.path.dirname(madgraph.__file__), 'iolibs', 'template_files',
                   'madmatrix', 'madmatrix.mk')
        with open(mk) as f:
            text = f.read()
        default = text[text.index('ifeq ($(FPTYPE),)'):]
        default = default[:default.index('endif')]
        self.assertIn('override FPTYPE = v', default)
        self.assertIn('override MADMATRIX_BUILDDIR = build.$(BACKEND)_$(FPTYPE)', text)

    def test_reweighting_fallbacks_are_v(self):
        from madgraph.iolibs.template_files.mg7 import param_reweight_card
        self.assertEqual(param_reweight_card.DEFAULT_FPTYPE, 'v')


class TestSummarize(unittest.TestCase):

    def test_statistics(self):
        summary = precision_check.summarize([1.0, 2.002, 3.0], [1.0, 2.0, 3.0], 1e-4)
        self.assertEqual(summary['events'], 3)
        self.assertEqual(summary['non_finite'], 0)
        self.assertAlmostEqual(summary['max_relative_deviation'], 1e-3)
        self.assertAlmostEqual(summary['mean_relative_deviation'], 1e-3 / 3)
        # signed mean of r = v/d - 1: the relative change of the cross section
        self.assertAlmostEqual(summary['cross_section_bias'], 1e-3 / 3)
        self.assertEqual(summary['above_threshold'], 1)
        self.assertEqual(summary['threshold'], 1e-4)

    def test_zero_and_non_finite(self):
        summary = precision_check.summarize(
            [0.0, 1.0, float('nan'), float('inf')], [0.0, 0.0, 1.0, 1.0], 1e-3)
        # 0 vs 0 agrees; 1 vs 0, nan and inf are all infinitely off
        self.assertEqual(summary['non_finite'], 2)
        self.assertEqual(summary['above_threshold'], 3)
        self.assertTrue(math.isinf(summary['max_relative_deviation']))
        # the bias and the mean |r| only average finite events with a
        # non-zero reference
        self.assertEqual(summary['cross_section_bias'], 0.0)
        self.assertEqual(summary['mean_relative_deviation'], 0.0)

    def test_one_bad_event_keeps_the_mean(self):
        summary = precision_check.summarize(
            [1.001, 0.999, float('nan')], [1.0, 1.0, 1.0], 1e-2)
        self.assertAlmostEqual(summary['mean_relative_deviation'], 1e-3)
        self.assertTrue(math.isinf(summary['max_relative_deviation']))
        self.assertEqual(summary['above_threshold'], 1)

    def test_bias_is_signed(self):
        summary = precision_check.summarize([1.001, 0.999], [1.0, 1.0], 1e-2)
        self.assertAlmostEqual(summary['cross_section_bias'], 0.0)
        self.assertAlmostEqual(summary['mean_relative_deviation'], 1e-3)

    def test_json_safe(self):
        import json
        summary = precision_check.summarize([float('inf')], [1.0], 1e-3)
        text = json.dumps(precision_check._json_safe(summary), allow_nan=False)
        self.assertEqual(json.loads(text)['max_relative_deviation'], 'inf')

    def test_identical(self):
        summary = precision_check.summarize([1.5, 2.5], [1.5, 2.5], 1e-3)
        self.assertEqual(summary['max_relative_deviation'], 0.0)
        self.assertEqual(summary['above_threshold'], 0)


class TestFlavorTable(unittest.TestCase):
    """Events are matched to a library and flavour by their PDG ids."""

    class _Subprocess(object):
        """The attributes of a MadgraphSubprocess the check reads."""
        def __init__(self, meta, unmerged_meta=None, subproc_id=0):
            self.meta = meta
            self.unmerged_meta = unmerged_meta
            self.subproc_id = subproc_id

    def _subprocess(self, meta, unmerged_meta=None, subproc_id=0):
        return self._Subprocess(meta, unmerged_meta, subproc_id)

    class _Process(object):
        def __init__(self, subprocesses):
            self.subprocesses = subprocesses

    def test_options_and_mirror(self):
        meta = {
            'incoming': [81, -81], 'outgoing': [-11, 11, 21],
            'me_path': 'lib/libA_{device}_{fptype}.so',
            'flavors': [
                {'index': 4, 'mirror': True,
                 'options': [[1, -1, -11, 11, 21], [3, -3, -11, 11, 21]]},
                {'index': 5, 'mirror': False, 'options': [[2, -2, -11, 11, 21]]},
            ],
        }
        table = precision_check.flavor_table(self._Process([self._subprocess(meta)]))
        self.assertEqual(table[(1, -1, -11, 11, 21)], ('lib/libA_{device}_{fptype}.so', 4, False))
        self.assertEqual(table[(3, -3, -11, 11, 21)], ('lib/libA_{device}_{fptype}.so', 4, False))
        # a mirrored flavour, reached with its incoming legs exchanged
        self.assertEqual(table[(-1, 1, -11, 11, 21)], ('lib/libA_{device}_{fptype}.so', 4, True))
        self.assertEqual(table[(2, -2, -11, 11, 21)], ('lib/libA_{device}_{fptype}.so', 5, False))
        self.assertNotIn((-2, 2, -11, 11, 21), table)

    def test_amplicol_subprocess_is_left_out(self):
        meta = {'incoming': [21, 21], 'outgoing': [21, 21], 'backend': 'amplicol',
                'me_path': 'lib/libAC.so',
                'flavors': [{'index': 0, 'mirror': False, 'options': [[21, 21, 21, 21]]}]}
        self.assertEqual(
            precision_check.flavor_table(self._Process([self._subprocess(meta)])), {})

    def test_merged_subprocess_members(self):
        unmerged = [
            {'incoming': [2, -2], 'outgoing': [23], 'me_path': 'lib/libU_{device}_{fptype}.so',
             'flavors': [{'index': 0, 'mirror': False, 'options': [[2, -2, 23]]}]},
            {'incoming': [1, -1], 'outgoing': [23], 'me_path': 'lib/libD_{device}_{fptype}.so',
             'flavors': [{'index': 0, 'mirror': False, 'options': [[1, -1, 23]]}]},
        ]
        merged = {'subprocesses': [0, 1],
                  'flavors': [{'subprocess': 0, 'flavor': 0}, {'subprocess': 1, 'flavor': 0}]}
        process = self._Process([self._subprocess(merged, unmerged, subproc_id=3)])
        table = precision_check.flavor_table(process)
        self.assertEqual(table[(2, -2, 23)], ('lib/libU_{device}_{fptype}.so', 0, False))
        self.assertEqual(table[(1, -1, 23)], ('lib/libD_{device}_{fptype}.so', 0, False))
        # compact_npy events name their subprocess and flavour instead
        self.assertEqual(precision_check._compact_pdgs(process, 3, 1), (1, -1, 23))
        self.assertIsNone(precision_check._compact_pdgs(process, 0, 0))

    def test_run_without_builds_checks_nothing(self):
        """No precision_check_builds (all64, precision_check = 0, a gridpack
        run, a failed build): nothing is checked, nothing is written."""
        class Process(object):
            precision_check_builds = None
            events_path = None
        self.assertIsNone(precision_check.run(Process()))


if __name__ == '__main__':
    unittest.main()
