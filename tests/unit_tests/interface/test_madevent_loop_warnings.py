"""Explicit MadLoop card choices must trigger the loop-induced diagnostics."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from madgraph.interface import madevent_interface


class TestLoopInducedCardWarnings(unittest.TestCase):
    def treat_card(self, nhel, settings):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Cards').mkdir()
            (root / 'SubProcesses' / 'MadLoop5_resources').mkdir(parents=True)
            (root / 'Cards' / 'MadLoopParams.dat').write_text(''.join(
                '#%s\n%s\n' % item for item in settings.items()))
            owner = SimpleNamespace(
                me_dir=directory,
                proc_characteristics={'loop_induced': True, 'nexternal': 4},
                run_card={'nhel': nhel}, check_param_card=mock.Mock())
            with mock.patch.object(madevent_interface.logger, 'warning') as warning, \
                    mock.patch.object(madevent_interface.logger, 'info') as info:
                madevent_interface.MadEventCmd.do_treatcards(
                    owner, '', mode='loop', opt={'forbid_MadLoopInit': True})
            return owner.MadLoopparam, [str(c.args[0]) for c in warning.call_args_list], \
                [str(c.args[0]) for c in info.call_args_list]

    def test_explicit_opp_warns_for_helicity_sum_and_keeps_choice(self):
        for library in ('1', '6', '8'):
            with self.subTest(library=library):
                card, warnings, _ = self.treat_card(0, {'MLReductionLib': library})
                self.assertEqual(card['MLReductionLib'], library)
                self.assertEqual(len(warnings), 1)
                self.assertIn('preferred reduction technique in MadLoop to be OPP', warnings[0])

    def test_explicit_tir_warns_for_helicity_sampling_and_keeps_choice(self):
        card, warnings, _ = self.treat_card(1, {'MLReductionLib': '7|8'})
        self.assertEqual(card['MLReductionLib'], '7|8')
        self.assertEqual(len(warnings), 1)
        self.assertIn('different than OPP', warnings[0])

    def test_appropriate_choices_and_implicit_defaults_do_not_warn(self):
        for nhel, settings in ((0, {}), (1, {}), (0, {'MLReductionLib': '7'}),
                               (1, {'MLReductionLib': '8'})):
            with self.subTest(nhel=nhel, settings=settings):
                card, warnings, _ = self.treat_card(nhel, settings)
                self.assertEqual(warnings, [])
                self.assertEqual(card['MLReductionLib'],
                                 settings.get('MLReductionLib', '7|6|1' if nhel == 0 else '6|7|1'))

    def test_other_explicit_card_diagnostics_use_normalized_names(self):
        card, warnings, info = self.treat_card(0, {
            'WriteOutFilters': '.true.', 'NRotations_QP': '1', 'MLStabThres': '1e-4'})
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any('lorentz rotation' in text for text in warnings))
        self.assertTrue(any('MLStabThres' in text for text in warnings))
        self.assertTrue(any('writing out filters' in text for text in info))
        self.assertTrue(card['WriteOutFilters'])
        self.assertEqual(card['NRotations_QP'], 1)
        self.assertEqual(card['MLStabThres'], 1e-4)
