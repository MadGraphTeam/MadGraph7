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
"""What ends up in the command history.

Two things a history file has to get right to be replayable with
`import command`: it must not end by rewriting itself, and it must contain the
answers given to questions -- a `set` typed at the launch card question is part
of the run, and leaving it out produced a file that silently reproduced
something else.
"""

from __future__ import absolute_import

import os
import shutil
import tempfile
import unittest

import madgraph.interface.extended_cmd as extended_cmd


class _Interface(extended_cmd.Cmd):
    """A Cmd with just the bits history touches."""

    def __init__(self):
        self.history = extended_cmd.HistoryList() \
            if hasattr(extended_cmd, 'HistoryList') else []
        self.mother = None
        self.log = False
        self._export_dir = None

    def split_arg(self, line):
        return line.split()

    def check_history(self, args):
        return True

    def get_history_header(self):
        return '# header\n'


class TestHistoryDoesNotWriteItself(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.interface = _Interface()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def written(self, name='out.dat'):
        path = os.path.join(self.dir, name)
        # precmd records the command before do_history runs, as it does live
        self.interface.history.append('history %s' % path)
        self.interface.do_history(path)
        with open(path) as handle:
            return [l for l in handle.read().split('\n')
                    if l and not l.startswith('#')]

    def test_the_history_command_is_not_in_the_file(self):
        self.interface.history.extend(['generate p p > t t~', 'output PROC'])
        lines = self.written()
        self.assertEqual(lines, ['generate p p > t t~', 'output PROC'])

    def test_everything_else_survives(self):
        self.interface.history.extend(['import model sm', 'generate p p > z'])
        self.assertIn('import model sm', self.written())

    def test_it_only_drops_a_trailing_history_command(self):
        """An earlier `history` is not the one being run."""

        self.interface.history.extend(['history first.dat',
                                       'generate p p > t t~'])
        lines = self.written()
        self.assertIn('history first.dat', lines)
        self.assertIn('generate p p > t t~', lines)

    def test_an_empty_history_is_fine(self):
        self.assertEqual(self.written(), [])


class TestQuestionAnswersAreRecorded(unittest.TestCase):
    """A question runs its own cmdloop, so what is typed at it never reached
    the history of the interface that asked."""

    def test_a_typed_answer_reaches_the_mother(self):
        mother = _Interface()
        question = extended_cmd.SmartQuestion('pick one',
                                              mother_interface=mother)
        question.precmd('set nevents 321')
        self.assertEqual(list(mother.history), ['set nevents 321'])

    def test_an_empty_answer_is_not_recorded(self):
        """Pressing Enter takes the default; an empty line in a command file
        is skipped on replay anyway, so recording it would add noise that
        cannot mean anything."""

        mother = _Interface()
        question = extended_cmd.SmartQuestion('pick one',
                                              mother_interface=mother)
        for blank in ('', '   ', None):
            question.precmd(blank)
        self.assertEqual(list(mother.history), [])

    def test_precmd_still_returns_the_line(self):
        mother = _Interface()
        question = extended_cmd.SmartQuestion('pick one',
                                              mother_interface=mother)
        self.assertEqual(question.precmd('done'), 'done')

    def test_it_records_up_the_mother_chain(self):
        """The question's mother is often a child interface -- the run
        interface `launch` created -- while `history` is typed at the one the
        user started from. Both need it.
        """

        root, child = _Interface(), _Interface()
        child.mother = root
        question = extended_cmd.SmartQuestion('pick one',
                                              mother_interface=child)
        question.precmd('set nevents 321')
        self.assertEqual(list(child.history), ['set nevents 321'])
        self.assertEqual(list(root.history), ['set nevents 321'])

    def test_a_cycle_in_the_chain_does_not_hang(self):
        a, b = _Interface(), _Interface()
        a.mother, b.mother = b, a
        extended_cmd.record_answer_in_history(a, 'done')
        self.assertEqual(list(a.history), ['done'])
        self.assertEqual(list(b.history), ['done'])

    def test_an_interface_without_a_history_is_tolerated(self):
        class _Bare(object):
            mother = None

        extended_cmd.record_answer_in_history(_Bare(), 'done')   # must not raise

    def test_no_mother_is_tolerated(self):
        question = extended_cmd.SmartQuestion('pick one', mother_interface=None)
        self.assertEqual(question.precmd('done'), 'done')
