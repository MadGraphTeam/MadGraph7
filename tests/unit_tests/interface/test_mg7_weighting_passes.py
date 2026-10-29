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
"""The [surrogate] and [amplicol] second passes of an mg7 run: the run card
settings of each pass and the checks done at launch."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from madgraph.mg7 import bootstrap as mg7_bootstrap


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class FirstPassSettingsTest(unittest.TestCase):
    """The first_pass_ settings apply to the subprocesses unweighted again in a
    second pass only, the [generation] ones to all others."""

    def setUp(self):
        import madgraph.various.banner as banner_mod
        from madgraph.mg7 import madgraph_process
        from madgraph.mg7.subprocess_amplicol import AmplicolSubprocess
        from madgraph.mg7.subprocess_madmatrix import MadgraphSubprocess
        self.ms = madgraph_process.ms
        self.process = object.__new__(madgraph_process.MadgraphProcess)
        run_card = banner_mod.RunCardMG7()
        run_card['generation']['max_overweight_truncation'] = 0.001
        run_card['generation']['freeze_max_weight_after'] = 500
        run_card['amplicol']['mode'] = 'unweight'
        run_card['amplicol']['first_pass_max_overweight_truncation'] = 0.02
        run_card['amplicol']['first_pass_freeze_max_weight_after'] = 70
        self.process.run_card = run_card
        self.process.surrogate = None
        self.process.subprocess_data = []
        step = self.ms.WeightingStep([], mode='unweight')

        amplicol = object.__new__(AmplicolSubprocess)
        amplicol.process = self.process
        amplicol.ratio_matrix_element = object()
        amplicol.weighting_steps = lambda: [step]
        madmatrix = object.__new__(MadgraphSubprocess)
        madmatrix.process = self.process
        madmatrix.surrogates = []
        self.process.subprocesses = [amplicol, madmatrix]

    def test_mixed_output(self):
        groups = self.process.weighting_groups([[0, 1], [2]])
        self.assertEqual(len(groups), 1)
        self.assertEqual(list(groups[0].channel_indices), [0, 1])
        self.assertAlmostEqual(groups[0].first_pass_max_overweight_truncation, 0.02)
        self.assertEqual(groups[0].first_pass_freeze_max_weight_after, 70)

        self.process.init_generator_config()
        cfg = self.process.event_generator_config
        self.assertAlmostEqual(cfg.max_overweight_truncation, 0.001)
        self.assertEqual(cfg.freeze_max_weight_after, 500)

    def test_reweight_mode_has_no_first_pass(self):
        self.process.run_card['amplicol']['mode'] = 'reweight'
        groups = self.process.weighting_groups([[0, 1], [2]])
        self.assertIsNone(groups[0].first_pass_max_overweight_truncation)
        self.assertIsNone(groups[0].first_pass_freeze_max_weight_after)


_TEMPLATES = (Path(__file__).resolve().parents[3] / 'madgraph' / 'iolibs'
              / 'template_files' / 'mg7')


def _template_module(name):
    spec = importlib.util.spec_from_file_location(
        f'mg7_template_{name}', _TEMPLATES / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class GridpackFirstPassTest(unittest.TestCase):
    """A gridpack takes the first_pass_ settings from its card, for the groups
    that unweight in a later step, also if its groups.json predates them."""

    def load(self, groups):
        setup = _template_module('gridpack_setup')
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, 'data'))
            with open(os.path.join(tmp, 'data', 'groups.json'), 'w') as f:
                json.dump(groups, f)
            with mock.patch.object(setup, 'gridpack_path',
                                   lambda *parts: os.path.join(tmp, *parts)):
                return setup.load_weighting_groups(0.001, 500, 0.02, 70)

    def test_old_and_new_groups(self):
        ms = _template_module('gridpack_setup').ms
        step = ms.WeightingStep([], mode='unweight')
        stored = json.loads(ms.weighting_groups_to_json([
            ms.WeightingGroup([0], [step], 0.5, 3), ms.WeightingGroup([1], [step]),
        ]))
        reweight = json.loads(ms.weighting_groups_to_json([
            ms.WeightingGroup([2], [ms.WeightingStep([], mode='reweight')], 0.5, 3)
        ]))
        groups = self.load(stored + reweight)
        for group in groups[:2]:
            self.assertAlmostEqual(group.first_pass_max_overweight_truncation, 0.02)
            self.assertEqual(group.first_pass_freeze_max_weight_after, 70)
            self.assertAlmostEqual(group.steps[0].max_overweight_truncation, 0.001)
        self.assertIsNone(groups[2].first_pass_max_overweight_truncation)


if __name__ == '__main__':
    unittest.main()
