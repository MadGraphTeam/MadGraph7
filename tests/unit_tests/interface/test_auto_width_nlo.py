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
"""NLO widths with MC@NLO, and the options of an 'auto' width.

``compute_widths --nlo`` runs the two-body decays that have a colour charge at NLO
with aMC@NLO (when the model has no dedicated code, SMWidth), keeps the others
and the decays into three bodies or more at LO. A width on auto in a param_card
takes the options of the command: ``DECAY 6 auto --nlo --body_decay=3``.

The runs of aMC@NLO are stubbed here (the real ones are in
tests/acceptance_tests/test_auto_width.py); the test of the launcher and the one
of the build need madspace and skip without it.
"""

from __future__ import absolute_import

import logging
import os
import shutil
import tempfile
import unittest
from unittest import mock

import aloha
import madgraph
import madgraph.interface.common_run_interface as common_run
import madgraph.interface.madevent_interface as madevent_interface
import madgraph.interface.master_interface as mgcmd
import models.check_param_card as check_param_card
import models.import_ufo as import_ufo
import models.model_reader as model_reader
from madgraph.iolibs.template_files.mg7 import bootstrap as mg7_bootstrap

pjoin = os.path.join


class AutoWidthOptionsTest(unittest.TestCase):
    """The options after the 'auto' of a width."""

    def test_plain_auto(self):
        self.assertEqual(check_param_card.auto_width_options('auto'), [])
        self.assertEqual(check_param_card.auto_width_options('Auto'), [])

    def test_the_older_nlo_spelling(self):
        self.assertEqual(check_param_card.auto_width_options('auto@NLO'), ['--nlo'])
        self.assertEqual(check_param_card.auto_width_options('Auto@nlo'), ['--nlo'])

    def test_options(self):
        self.assertEqual(check_param_card.auto_width_options(
            'auto --nlo --body_decay=3'), ['--nlo', '--body_decay=3'])
        self.assertEqual(check_param_card.auto_width_options(
            'AUTO --engine=madevent --precision_channel=0.001 --min_br=0.01'),
            ['--precision_channel=0.001', '--min_br=0.01', '--engine=madevent'])

    def test_one_order_whatever_the_user_wrote(self):
        self.assertEqual(
            check_param_card.auto_width_options('auto --body_decay=3 --nlo'),
            check_param_card.auto_width_options('auto --nlo --body_decay=3'))

    def test_a_repeated_option_is_one_option(self):
        self.assertEqual(check_param_card.auto_width_options(
            'auto@NLO --nlo --body_decay=2 --body_decay=3'), ['--nlo', '--body_decay=3'])

    def test_not_auto(self):
        for value in (0.5, 3, '0.5', 'scan:[1,2]', 'autox', '', None):
            self.assertIsNone(check_param_card.auto_width_options(value))

    def test_invalid_options(self):
        for value in ('auto --foo', 'auto nlo', 'auto --body_decay', 'auto --body_decay=',
                      'auto --nlo=1', 'auto --path=card.dat'):
            with self.assertRaises(check_param_card.InvalidParam, msg=value):
                check_param_card.auto_width_options(value)

    def test_the_text_of_the_card(self):
        self.assertEqual(check_param_card.format_auto_width('auto'), 'Auto')
        self.assertEqual(check_param_card.format_auto_width('auto@nlo'), 'Auto@NLO')
        self.assertEqual(check_param_card.format_auto_width('auto --nlo'), 'Auto@NLO')
        self.assertEqual(check_param_card.format_auto_width('auto --body_decay=3 --nlo'),
                         'Auto --nlo --body_decay=3')

    def test_groups_of_widths(self):
        text = '\n'.join([
            'DECAY  6 auto # WT',
            'DECAY  25 Auto --nlo --body_decay=3 # WH',
            'DECAY  23 auto@NLO',
            'DECAY  24 2.0',
            'DECAY  -6 auto',
            'decay 1000021 auto --body_decay=3 --nlo',
            'DECAY  15 0.0 # no',
        ])
        groups = check_param_card.group_auto_widths(text)
        self.assertEqual(dict(groups), {
            (): ['6'],
            ('--nlo', '--body_decay=3'): ['25', '1000021'],
            ('--nlo',): ['23']})
        # in the order of the card
        self.assertEqual(list(groups), [(), ('--nlo', '--body_decay=3'), ('--nlo',)])

    def test_no_auto_width(self):
        self.assertEqual(len(check_param_card.group_auto_widths('DECAY 6 1.5\n')), 0)

    def test_an_invalid_option_in_the_card(self):
        with self.assertRaises(check_param_card.InvalidParam):
            check_param_card.group_auto_widths('DECAY 6 auto --foo\n')


class AutoWidthCardTest(unittest.TestCase):
    """The card keeps and rewrites the widths on auto."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='test_auto_width_nlo')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, name, text):
        path = pjoin(self.tmp, name)
        with open(path, 'w') as stream:
            stream.write(text)
        return path

    def test_update_keeps_the_widths_still_on_auto(self):
        """compute_widths computes the widths with the same options together:
        the others stay on auto in the card, they used to be dropped."""
        path = self.write('param_card.dat', '\n'.join([
            'Block mass', '    6 1.730000e+02 # MT', '    23 9.118800e+01 # MZ',
            'DECAY  6 auto --nlo --body_decay=3 # WT',
            'DECAY  23 2.4 # WZ',
            'DECAY  24 Auto@NLO # WW',
            'DECAY  25 auto # WH', '']))
        out = pjoin(self.tmp, 'out.dat')
        madevent_interface.MadEventCmd.update_width_in_param_card(
            {25: [[[5, -5], 4e-3]]}, path, out)
        with open(out) as stream:
            text = stream.read()
        self.assertIn('DECAY  6  auto --nlo --body_decay=3 #', text)
        self.assertIn('DECAY  24  Auto@NLO #', text)
        self.assertEqual(check_param_card.group_auto_widths(text),
                         check_param_card.group_auto_widths(
                             'DECAY 6 auto --nlo --body_decay=3\nDECAY 24 auto@NLO'))
        card = check_param_card.ParamCard(out)
        self.assertAlmostEqual(card['decay'].get((25,)).value, 4e-3, places=9)
        self.assertAlmostEqual(card['decay'].get((23,)).value, 2.4)

    def test_the_card_reads_the_options_back(self):
        path = self.write('param_card.dat', '\n'.join([
            'Block mass', '    6 1.730000e+02 # MT',
            'DECAY  6 auto --nlo --body_decay=3 # WT', '']))
        card = check_param_card.ParamCard(path)
        self.assertEqual(card['decay'].get((6,)).value, 'auto --nlo --body_decay=3')
        out = pjoin(self.tmp, 'out.dat')
        card.write(out)
        with open(out) as stream:
            self.assertIn('DECAY 6 auto --nlo --body_decay=3', stream.read())

    def test_merge_takes_the_values_of_the_user(self):
        source = self.write('source.dat', '\n'.join([
            'Block mass', '    6 1.700000e+02 # MT', '    25 1.260000e+02 # MH',
            '    9999 1.0 # not in the target',
            'Block unknown', '    1 5.0 # nowhere',
            'DECAY  6 auto # WT', 'DECAY  25 4.1e-3 # WH', '']))
        target = self.write('target.dat', '\n'.join([
            'Block mass', '    6 1.730000e+02 # MT', '    25 1.250000e+02 # MH',
            'Block loop', '    1 9.118800e+01 # MU_R',
            'DECAY  6 1.5 # WT', 'DECAY  25 6.0e-3 # WH', '']))
        check_param_card.merge_param_card_values(source, target)
        card = check_param_card.ParamCard(target)
        self.assertEqual(card['mass'].get((6,)).value, 170.)
        self.assertEqual(card['mass'].get((25,)).value, 126.)
        self.assertEqual(card['loop'].get((1,)).value, 91.188)
        self.assertNotIn('unknown', card)
        # a width still on auto is not a number to take
        self.assertEqual(card['decay'].get((6,)).value, 1.5)
        self.assertAlmostEqual(card['decay'].get((25,)).value, 4.1e-3)

    def test_the_options_stay_in_the_value_of_set(self):
        join = common_run.AskforEditCard.join_auto_width_options
        self.assertEqual(join(['decay', '6', 'auto', '--nlo', '--body_decay=3']),
                         ['decay', '6', 'auto --nlo --body_decay=3'])
        self.assertEqual(join(['decay', 'all', 'AUTO@NLO', '--body_decay=2']),
                         ['decay', 'all', 'AUTO@NLO --body_decay=2'])
        for args in (['decay', '6', 'auto'], ['decay', '6', 'auto@nlo'],
                     ['mass', '6', '172.5'], ['decay', '6', '1.5'], ['auto']):
            self.assertEqual(join(list(args)), args)

    def test_set_decay_auto_with_options(self):
        """setP writes the canonical text of the width"""
        editor = object.__new__(common_run.AskforEditCard)
        card = check_param_card.ParamCard(self.write('param_card.dat', '\n'.join([
            'Block mass', '    6 1.730000e+02 # MT', 'DECAY  6 1.5 # WT', ''])))
        editor.param_card = card
        for value, expected in (('auto', 'Auto'), ('auto@nlo', 'Auto@NLO'),
                                ('auto --body_decay=3 --nlo', 'Auto --nlo --body_decay=3')):
            editor.setP('decay', (6,), value)
            self.assertEqual(card['decay'].get((6,)).value, expected)

    def test_set_decay_auto_with_a_wrong_option(self):
        editor = object.__new__(common_run.AskforEditCard)
        card = check_param_card.ParamCard(self.write('param_card.dat', '\n'.join([
            'Block mass', '    6 1.730000e+02 # MT', 'DECAY  6 1.5 # WT', ''])))
        editor.param_card = card
        editor.setP('decay', (6,), 'auto --foo')
        self.assertEqual(card['decay'].get((6,)).value, 1.5)

    def test_one_computation_per_set_of_options(self):
        path = self.write('param_card.dat', '\n'.join([
            'Block mass', '    6 1.730000e+02 # MT', '    23 9.118800e+01 # MZ',
            '    25 1.250000e+02 # MH',
            'DECAY  6 auto --nlo --body_decay=3 # WT',
            'DECAY  23 auto@NLO # WZ',
            'DECAY  25 auto # WH',
            'DECAY  24 auto --nlo --body_decay=3 # WW', '']))
        lines = []

        def compute(line, interface, card=None):
            lines.append(line)
            # what compute_widths does to the card: numbers instead of auto
            if len(lines) == len(expected):
                with open(path) as stream:
                    text = stream.read()
                with open(path, 'w') as stream:
                    stream.write(text.replace('auto --nlo --body_decay=3', '1.0')
                                 .replace('auto@NLO', '1.0').replace('auto ', '1.0 '))

        expected = ['6 24 --nlo --body_decay=3', '23 --nlo', '25']
        with mock.patch.object(common_run.CommonRunCmd, 'static_compute_widths',
                               staticmethod(compute)):
            common_run.CommonRunCmd.static_check_param_card(path, mock.Mock(), run=True)
        self.assertEqual(lines, expected)


class NLOWidthsTest(unittest.TestCase):
    """compute_widths --nlo: which decays are run, and what becomes of the others."""

    @classmethod
    def setUpClass(cls):
        cls.cmd = mgcmd.MasterCmd()
        cls.cmd.no_notification()
        cls.cmd.exec_cmd('import model sm')
        cls.model = model_reader.ModelReader(import_ufo.import_model(
            cls.cmd._curr_model.get('modelpath+restriction'), decay=True))

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='test_auto_width_nlo')
        self.card = pjoin(self.tmp, 'param_card.dat')
        with open(self.card, 'w') as stream:
            stream.write(self.cmd._curr_model.write_param_card())
        self.opts = {'path': self.card, 'output': self.card, 'precision_channel': 0.01}
        self.runs = []
        self.updated = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    #: the LO widths of a computation up to three bodies
    LO = {6: [[[24, 5], 1.49]],
          23: [[[-1, 1], 0.372], [[-3, 3], 0.372], [[-2, 2], 0.290], [[-4, 4], 0.290],
               [[-5, 5], 0.368], [[-11, 11], 0.084], [[-12, 12], 0.166],
               [[-24, -1, 2], 1e-7]],
          25: [[[5, -5], 5.39e-3], [[-15, 15], 2.6e-4], [[-24, 1, -2], 1.3e-4]]}

    def born_orders(self, pids, opts, decaymodel=None):
        orders = {}
        for pid in pids:
            for decay_to, _ in self.LO[pid]:
                orders[(pid, tuple(sorted(decay_to)))] = {'QED': 1}
        return orders

    def run_nlo(self, k_factor=0.9, fail=(), decay_info=None):
        """compute_widths_nlo with the aMC@NLO run stubbed (its width is the
        k_factor times the LO width, read off the process)."""
        decay_info = decay_info or self.LO
        lo_of = {}
        for pid, channels in decay_info.items():
            for decay_to, width in channels:
                name = ' '.join(self.cmd.particle_name(self.model, code)
                                for code in [pid] + decay_to)
                lo_of[name.replace(' ', ' > ', 1)] = width

        def run(workdir, process, model_ref, param_card, precision, nb_core):
            self.runs.append((process, model_ref, param_card, precision))
            decay = process.split(' QCD')[0].split(' QED')[0]
            if any(name in decay for name in fail):
                raise madgraph.MadGraph5Error('the run failed')
            return lo_of[decay] * k_factor, lo_of[decay] * 0.001

        with mock.patch.object(self.cmd, 'run_mcatnlo_width', run), \
             mock.patch.object(self.cmd, 'two_body_born_orders', self.born_orders), \
             mock.patch.object(madevent_interface.MadEventCmd,
                               'update_width_in_param_card',
                               lambda info, initial=None, output=None:
                                   self.updated.append((info, initial, output))):
            return self.cmd.compute_widths_nlo(sorted(decay_info), self.opts,
                                               decay_info, self.model)

    def test_only_the_decays_with_a_colour_charge_are_run(self):
        self.run_nlo()
        processes = sorted(process.split(' QCD')[0] for process, _, _, _ in self.runs)
        # t > w+ b, h > b b~, the quarks of the Z (one run per kind of quark)
        self.assertEqual(len([p for p in processes if p.startswith('t > ')]), 1)
        self.assertEqual(len([p for p in processes if p.startswith('h > ')]), 1)
        self.assertFalse([p for p in processes if 'ta' in p or ' e-' in p or ' ve' in p])
        for process, _, _, _ in self.runs:
            self.assertTrue(process.endswith('[QCD]'))
            self.assertIn('QED=1', process)
            self.assertIn('QCD=0', process)

    def test_flavours_treated_alike_are_run_once(self):
        """d and s of the Z have the same LO width and the same masses: one run;
        u and c as well; the b is another run."""
        self.run_nlo()
        z_runs = [process for process, _, _, _ in self.runs if process.startswith('z > ')]
        self.assertEqual(len(z_runs), 3)
        self.assertEqual(len(self.runs), 5)

    def test_the_nlo_widths_replace_the_lo_ones(self):
        decay_info = self.run_nlo(k_factor=0.9)
        widths = dict((tuple(decay_to), width) for decay_to, width in decay_info[23])
        self.assertAlmostEqual(widths[(-1, 1)], 0.372 * 0.9)
        self.assertAlmostEqual(widths[(-3, 3)], 0.372 * 0.9)
        self.assertAlmostEqual(widths[(-4, 4)], 0.290 * 0.9)
        self.assertAlmostEqual(widths[(-5, 5)], 0.368 * 0.9)
        self.assertAlmostEqual(decay_info[6][0][1], 1.49 * 0.9)
        self.assertAlmostEqual(decay_info[25][0][1], 5.39e-3 * 0.9)

    def test_the_others_stay_at_lo(self):
        decay_info = self.run_nlo(k_factor=0.9)
        widths = dict((tuple(decay_to), width) for decay_to, width in decay_info[23])
        self.assertEqual(widths[(-11, 11)], 0.084)        # no colour
        self.assertEqual(widths[(-12, 12)], 0.166)
        self.assertEqual(widths[(-24, -1, 2)], 1e-7)      # three bodies
        higgs = dict((tuple(decay_to), width) for decay_to, width in decay_info[25])
        self.assertEqual(higgs[(-15, 15)], 2.6e-4)
        self.assertEqual(higgs[(-24, 1, -2)], 1.3e-4)

    def test_the_lo_decay_info_is_not_modified(self):
        reference = [[list(decay_to), width] for decay_to, width in self.LO[23]]
        self.run_nlo()
        self.assertEqual(self.LO[23], reference)

    def test_the_card_is_rewritten_with_the_sums(self):
        decay_info = self.run_nlo()
        self.assertEqual(len(self.updated), 1)
        info, initial, output = self.updated[0]
        self.assertEqual(info, decay_info)
        self.assertEqual((initial, output), (self.card, self.card))

    def test_a_decay_that_fails_keeps_its_lo_width(self):
        warnings = []
        handler = logging.Handler()
        handler.emit = lambda record: warnings.append(record.getMessage())
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger('cmdprint')
        logger.addHandler(handler)
        try:
            decay_info = self.run_nlo(k_factor=0.9, fail=('t > ',))
        finally:
            logger.removeHandler(handler)
        self.assertEqual(decay_info[6][0][1], 1.49)
        self.assertAlmostEqual(decay_info[25][0][1], 5.39e-3 * 0.9)
        self.assertTrue([w for w in warnings if 't > w+ b' in w and 'LO' in w], warnings)

    def test_nothing_to_do(self):
        info = {23: [[[-11, 11], 0.084], [[-12, 12], 0.166]],
                25: [[[-15, 15], 2.6e-4]]}
        decay_info = self.run_nlo(decay_info=info)
        self.assertEqual(decay_info, info)
        self.assertFalse(self.runs)
        self.assertFalse(self.updated)

    def test_a_decay_without_width_is_not_run(self):
        info = {25: [[[5, -5], 0.0]]}
        self.run_nlo(decay_info=info)
        self.assertFalse(self.runs)

    def test_the_precision_and_the_model_are_passed_on(self):
        self.opts['precision_channel'] = 0.005
        self.run_nlo()
        for _, model_ref, param_card, precision in self.runs:
            self.assertEqual(precision, 0.005)
            self.assertEqual(model_ref, self.cmd._curr_model.get('modelpath+restriction'))
            self.assertEqual(param_card, self.card)

    def test_names_of_particles(self):
        name = self.cmd.particle_name
        self.assertEqual(name(self.model, 24), 'w+')
        self.assertEqual(name(self.model, -24), 'w-')
        self.assertEqual(name(self.model, -5), 'b~')
        self.assertEqual(name(self.model, 23), 'z')

    # ------------------------------------------------------------------ dispatch

    def run_compute_widths(self, line, smwidth):
        with mock.patch.object(self.cmd, 'has_smwidth', lambda model: smwidth), \
             mock.patch.object(self.cmd, 'compute_widths_SMWidth',
                               mock.Mock(return_value={6: [[[], 1.3]]})) as dedicated, \
             mock.patch.object(self.cmd, '_compute_widths_lo',
                               mock.Mock(return_value=({6: [[[24, 5], 1.49]]}, [6],
                                                       self.opts))) as lo, \
             mock.patch.object(self.cmd, 'compute_widths_nlo',
                               mock.Mock(return_value={6: [[[24, 5], 1.35]]})) as nlo:
            result = self.cmd.do_compute_widths('%s --path=%s' % (line, self.card))
        return result, dedicated, lo, nlo

    def test_a_model_with_a_dedicated_code_uses_it(self):
        result, dedicated, lo, nlo = self.run_compute_widths('t --nlo', smwidth=True)
        self.assertEqual(result, {6: [[[], 1.3]]})
        dedicated.assert_called_once()
        lo.assert_not_called()
        nlo.assert_not_called()

    def test_the_others_run_the_two_body_decays_at_nlo_on_top_of_lo(self):
        result, dedicated, lo, nlo = self.run_compute_widths(
            't --nlo --body_decay=3', smwidth=False)
        dedicated.assert_not_called()
        lo.assert_called_once()
        nlo.assert_called_once()
        self.assertEqual(nlo.call_args[0][2], {6: [[[24, 5], 1.49]]})     # the LO widths
        self.assertEqual(result, {6: [[[24, 5], 1.35]]})

    def test_without_nlo_nothing_changes(self):
        result, dedicated, lo, nlo = self.run_compute_widths('t', smwidth=True)
        dedicated.assert_not_called()
        lo.assert_called_once()
        nlo.assert_not_called()
        self.assertEqual(result, {6: [[[24, 5], 1.49]]})

    def test_the_model_is_checked_for_smwidth(self):
        model = mock.Mock()
        model.get.return_value = self.tmp
        self.assertFalse(self.cmd.has_smwidth(model))
        os.mkdir(pjoin(self.tmp, 'SMWidth'))
        self.assertTrue(self.cmd.has_smwidth(model))


class FakeAMCatNLO(object):
    """What the two runs of run_mcatnlo_width do to their directory: the first
    exports the process, the second writes the summary."""

    RUN_CARD = ' 10000 = nevents ! events\n 0.01   = req_acc_FO       ! accuracy\n 5 = other\n'
    PARAM_CARD = ('Block mass\n    6 1.730000e+02 # MT\n    5 4.700000e+00 # MB\n'
                  'Block loop\n    1 9.118800e+01 # MU_R\nDECAY  6 1.5 # WT\n')

    def __init__(self, test, summary='(Partial) decay width: 1.356e+00 +- 8.7e-03 GeV',
                 generate_failure=None, launch_failure=None):
        self.test = test
        self.summary = summary
        self.generate_failure = generate_failure
        self.launch_failure = launch_failure
        self.commands = []

    def __call__(self, command, cwd=None, stdout=None, stderr=None, stdin=None):
        script = command[-1]
        with open(script) as stream:
            text = stream.read()
        self.commands.append((os.path.basename(script), text, cwd))
        if script.endswith('generate.mg5'):
            proc = pjoin(cwd, 'PROC_NLO')
            os.makedirs(pjoin(proc, 'Cards'))
            with open(pjoin(proc, 'Cards', 'run_card.dat'), 'w') as stream:
                stream.write(self.RUN_CARD)
            with open(pjoin(proc, 'Cards', 'param_card.dat'), 'w') as stream:
                stream.write(self.PARAM_CARD)
            if self.generate_failure:
                stdout.write(self.generate_failure)
                return 0
        else:
            proc = pjoin(cwd, 'PROC_NLO')
            if self.launch_failure:
                stdout.write(self.launch_failure)
                return 1
            if self.summary is not None:
                os.makedirs(pjoin(proc, 'Events', 'run_01'))
                with open(pjoin(proc, 'Events', 'run_01', 'summary.txt'), 'w') as stream:
                    stream.write('   Final results\n      ' + self.summary + '\n')
        return 0


class RunMCatNLOTest(unittest.TestCase):
    """run_mcatnlo_width: the two runs, the card, the accuracy, the result."""

    @classmethod
    def setUpClass(cls):
        cls.cmd = mgcmd.MasterCmd()
        cls.cmd.no_notification()

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='test_auto_width_nlo')
        self.card = pjoin(self.tmp, 'user_card.dat')
        with open(self.card, 'w') as stream:
            stream.write('Block mass\n    6 1.700000e+02 # MT\nDECAY  6 auto # WT\n')
        self.workdir = pjoin(self.tmp, 'nlo_0')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_width(self, fake, precision=0.002):
        with mock.patch('subprocess.call', fake):
            return self.cmd.run_mcatnlo_width(
                self.workdir, 't > w+ b QCD=0 QED=1 [QCD]', '/models/sm', self.card,
                precision, 4)

    def test_the_width_and_its_error(self):
        fake = FakeAMCatNLO(self)
        width, error = self.run_width(fake)
        self.assertEqual((width, error), (1.356, 8.7e-3))

    def test_the_process_is_generated_and_exported(self):
        fake = FakeAMCatNLO(self)
        self.run_width(fake)
        name, text, cwd = fake.commands[0]
        self.assertEqual(name, 'generate.mg5')
        self.assertEqual(cwd, self.workdir)
        self.assertIn('import model /models/sm\n', text)
        self.assertIn('generate t > w+ b QCD=0 QED=1 [QCD]\n', text)
        self.assertIn('output %s -f\n' % pjoin(self.workdir, 'PROC_NLO'), text)
        self.assertIn('set nb_core 4 --no_save\n', text)
        self.assertNotIn('complex_mass_scheme', text)

    def test_it_runs_at_fixed_order_nlo(self):
        fake = FakeAMCatNLO(self)
        self.run_width(fake)
        name, text, cwd = fake.commands[1]
        self.assertEqual(name, 'launch.txt')
        self.assertIn('launch NLO -f\n', text)
        self.assertIn('set nb_core 4 --no_save\n', text)

    def test_the_card_has_the_parameters_of_the_user(self):
        fake = FakeAMCatNLO(self)
        self.run_width(fake)
        card = check_param_card.ParamCard(pjoin(self.workdir, 'PROC_NLO', 'Cards',
                                                'param_card.dat'))
        self.assertEqual(card['mass'].get((6,)).value, 170.)     # the user's
        self.assertEqual(card['mass'].get((5,)).value, 4.7)      # the model's
        self.assertEqual(card['loop'].get((1,)).value, 91.188)

    def test_the_accuracy_of_the_run_card(self):
        fake = FakeAMCatNLO(self)
        self.run_width(fake, precision=0.002)
        with open(pjoin(self.workdir, 'PROC_NLO', 'Cards', 'run_card.dat')) as stream:
            lines = stream.read().splitlines()
        self.assertIn(' 0.002   = req_acc_FO       ! accuracy', [l.rstrip() for l in lines])
        self.assertIn(' 10000 = nevents ! events', lines)
        self.assertIn(' 5 = other', lines)

    def test_the_complex_mass_scheme_is_passed_on(self):
        fake = FakeAMCatNLO(self)
        with mock.patch.object(aloha, 'complex_mass', True):
            self.run_width(fake)
        self.assertIn('set complex_mass_scheme True\n', fake.commands[0][1])

    def test_a_process_that_cannot_be_generated(self):
        fake = FakeAMCatNLO(self, generate_failure=
            'Error detected in sub-command generate t > w+ b [QCD]\n'
            'The current model heft does not allow to generate loop corrections of type QCD.\n')
        with self.assertRaises(madgraph.MadGraph5Error) as caught:
            self.run_width(fake)
        self.assertIn('loop corrections', str(caught.exception))
        self.assertEqual(len(fake.commands), 1)         # no run of the missing process

    def test_a_failed_run(self):
        fake = FakeAMCatNLO(self, launch_failure='Fortran runtime error: boom\n')
        with self.assertRaises(madgraph.MadGraph5Error) as caught:
            self.run_width(fake)
        self.assertIn('boom', str(caught.exception))

    def test_no_width_in_the_results(self):
        fake = FakeAMCatNLO(self, summary=None)
        with self.assertRaises(madgraph.MadGraph5Error) as caught:
            self.run_width(fake)
        self.assertIn('no width', str(caught.exception))


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class LauncherTest(unittest.TestCase):
    """The mg7 launcher: the widths on auto with their options, the build."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='test_auto_width_nlo')
        self.cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_one_compute_widths_per_set_of_options(self):
        from madgraph.iolibs.template_files.mg7 import launch as mg7_launch
        card = pjoin(self.tmp, 'param_card.dat')
        with open(card, 'w') as stream:
            stream.write('\n'.join([
                'DECAY  6 auto --nlo --body_decay=3 # WT', 'DECAY  23 auto@NLO',
                'DECAY  24 auto', 'DECAY  25 auto --nlo --body_decay=3', '']))
        scripts = []

        def run(command, *args, **kwargs):
            with open(command[-1]) as stream:
                scripts.append(stream.read())
            return mock.Mock(returncode=0)

        with mock.patch.object(mg7_launch, 'read_stored_model', lambda: 'sm'), \
             mock.patch('subprocess.run', run):
            mg7_launch.compute_auto_widths(card)
        self.assertEqual(len(scripts), 1)
        lines = scripts[0].splitlines()
        self.assertEqual(lines[0], 'import model sm')
        commands = [l.replace(card, 'CARD') for l in lines[1:]]
        self.assertEqual(commands, [
            'compute_widths 6 25 --path=CARD --nlo --body_decay=3',
            'compute_widths 23 --path=CARD --nlo',
            'compute_widths 24 --path=CARD '])

    def test_no_auto_no_run(self):
        from madgraph.iolibs.template_files.mg7 import launch as mg7_launch
        card = pjoin(self.tmp, 'param_card.dat')
        with open(card, 'w') as stream:
            stream.write('DECAY  6 1.5\n')
        with mock.patch('subprocess.run') as run:
            mg7_launch.compute_auto_widths(card)
        run.assert_not_called()

    def compile_args(self, subprocesses, extra=()):
        """The arguments of the make that builds ``subprocesses`` directories."""
        from madgraph.iolibs.template_files.mg7 import launch as mg7_launch
        process = object.__new__(mg7_launch.MadgraphProcess)
        process.run_card = {'run': {'device': ['cpu'], 'cpu_mode': 'simd_128',
                                    'precision': 'color32', 'cpu_thread_pool_size': 2}}
        process.run_path = self.tmp
        process.extra_make_args = list(extra)
        process.subprocess_data = [
            {'path': pjoin('SubProcesses', 'P%d_x' % i),
             'me_path': pjoin(self.tmp, 'libmadmatrix_P%d_x_{device}.so' % i)}
            for i in range(subprocesses)]
        with mock.patch.object(mg7_launch.MadgraphProcess, 'make_subprocesses') as make:
            process.compile_matrix_elements()
        make.assert_called_once()
        return make.call_args[0][1]

    def test_the_headers_are_precompiled_from_four_subprocesses(self):
        for subprocesses, expected in ((1, False), (3, False), (4, True), (25, True)):
            args = self.compile_args(subprocesses)
            self.assertEqual('PCH=1' in args, expected, '%d subprocesses' % subprocesses)
            self.assertIn('BACKEND=simd_128', args)

    def test_only_what_is_to_build_counts(self):
        """Six directories, but the libraries of three of them exist: below the
        threshold."""
        from madgraph.iolibs.template_files.mg7 import launch as mg7_launch
        for i in range(3):
            open(pjoin(self.tmp, 'libmadmatrix_P%d_x_simd_128.so' % i), 'w').close()
        process = object.__new__(mg7_launch.MadgraphProcess)
        process.run_card = {'run': {'device': ['cpu'], 'cpu_mode': 'simd_128',
                                    'precision': 'color32', 'cpu_thread_pool_size': 2}}
        process.run_path = self.tmp
        process.extra_make_args = []
        process.subprocess_data = [
            {'path': pjoin('SubProcesses', 'P%d_x' % i),
             'me_path': pjoin(self.tmp, 'libmadmatrix_P%d_x_{device}.so' % i)}
            for i in range(6)]
        with mock.patch.object(mg7_launch.MadgraphProcess, 'make_subprocesses') as make:
            process.compile_matrix_elements()
        self.assertNotIn('PCH=1', make.call_args[0][1])

    def test_an_explicit_pch_is_respected(self):
        self.assertNotIn('PCH=1', self.compile_args(10, ['PCH=0']))
        self.assertIn('PCH=0', self.compile_args(10, ['PCH=0']))
        self.assertEqual(self.compile_args(2, ['PCH=1']).count('PCH=1'), 1)
        self.assertEqual(self.compile_args(10, ['PCH=1']).count('PCH=1'), 1)
