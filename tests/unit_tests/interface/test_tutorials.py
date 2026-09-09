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
"""Tests for the tutorial mode.

The important one is TestPortedTutorials: it pins the text the three ported
tutorials emit for a given command to what the pre-2026 command-name lookup in
CmdExtended.postcmd emitted, so the port cannot drift.  When the text is later
inlined into the tutorial modules (and tutorial_text*.py deleted) this test is
what makes that safe.
"""

from __future__ import absolute_import

import logging
import re
import unittest

import madgraph
import madgraph.interface.madgraph_interface as mg_interface
import madgraph.interface.tutorials as tutorials
import madgraph.interface.tutorials.mixin as tutorial_mixin
import madgraph.interface.tutorial_text_nlo as legacy_nlo
import madgraph.interface.tutorial_text_madloop as legacy_madloop

from madgraph.interface.tutorials._port import retarget
from madgraph.interface.tutorials.session import Step, Tutorial, TutorialSession


class _Capture(logging.Handler):
    """Collect what the tutorial logger emits."""

    def __init__(self):
        logging.Handler.__init__(self)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class _BareInterface(mg_interface.CmdExtended):
    """A CmdExtended that skips the (expensive) real __init__.

    postcmd only needs the instance to answer hasattr(self, 'post_XXX'), so
    this is enough to drive the tutorial hook.
    """

    def __init__(self):
        pass


class _TutorialTestCase(unittest.TestCase):

    def setUp(self):
        self.capture = _Capture()
        self.logger = logging.getLogger('tutorial')
        self._saved = (self.logger.handlers, self.logger.propagate,
                       self.logger.level)
        self.logger.handlers = [self.capture]
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)

    def tearDown(self):
        (self.logger.handlers, self.logger.propagate,
         self.logger.level) = self._saved

    def run_lines(self, tutorial_name, lines):
        """Feed command lines to a fresh session; return [(line, [text])]."""

        interface = _BareInterface()
        tutorial_mixin.attach(interface, tutorials.start(tutorial_name))
        out = []
        for line in lines:
            self.capture.messages = []
            interface.postcmd(True, line)
            out.append((line, list(self.capture.messages)))
        return out


#===============================================================================
# the port is faithful
#===============================================================================

# every command line the pre-2026 lookup could resolve, plus a few that it
# deliberately could not
COMMAND_LINES = [
    'generate p p > t t~',
    'add process p p > W+ j, W+ > l+ vl @2',
    'display processes',
    'display diagrams',
    'output MY_FIRST_MG5_RUN',
    'open index.html',
    'launch MY_FIRST_MG5_RUN',
    'import model MSSM_SLHA2',
    'import model_v4 sm',
    'customize_model',
    'define v = w+ w- z a',
    'history my_mg5_cmd.dat',
    'open ./my_mg5_cmd.dat',
    'check',
    'check profile p p > t t~',
    'tutorial',
    'set gauge Feynman',
    '',
    'quit',
]


def _legacy_text(module, line):
    """What CmdExtended.postcmd used to print for `line`, or None.

    A transcription of the pre-2026 lookup, kept here on purpose: it is the
    specification the port has to meet.  `retarget` applies the one change the
    port is allowed to make -- the stale MG5_aMC> prompt -- so that any other
    drift still fails.
    """

    args = line.split()
    if not args:
        return None
    if len(args) == 1:
        keys = [args[0]]
    else:
        keys = [args[0] + '_' + args[1].split('.')[0], args[0]]
    for key in keys:
        text = getattr(module, key, None)
        if isinstance(text, str):
            return retarget(text).replace('\n', '\n\t')
    return None


class TestPortedTutorials(_TutorialTestCase):
    """`nlo` and `madloop` emit exactly what the old lookup emitted.

    `lo` is deliberately absent: it was rewritten, not ported (see the plan in
    docs/tutorial-mode-plan.md), so it has TestLoTutorial below instead.
    """

    def _check(self, name, legacy_module):
        for line, emitted in self.run_lines(name, COMMAND_LINES):
            expected = _legacy_text(legacy_module, line)
            if expected is None:
                self.assertEqual(emitted, [],
                                 'tutorial %s should stay silent on %r, said %r'
                                 % (name, line, emitted))
            else:
                self.assertEqual(emitted, [expected],
                                 'tutorial %s drifted on %r' % (name, line))

    def test_nlo_matches_legacy(self):
        self._check('nlo', legacy_nlo)

    def test_madloop_matches_legacy(self):
        self._check('madloop', legacy_madloop)

    def test_no_stale_prompt_anywhere(self):
        """No tutorial still tells the user to type at an MG5_aMC> prompt."""

        for tutorial in tutorials.all_tutorials(include_hidden=True):
            for step in tutorial.steps:
                text = step.render(None)
                for stale in ('MG5_aMC>', 'MG_aMC>'):
                    self.assertNotIn(
                        stale, text,
                        'tutorial %s, step %r still uses the %s prompt'
                        % (tutorial.name, step.title, stale))


class TestLoTutorial(_TutorialTestCase):
    """The rewritten `lo`: a short, sequenced path to an event sample."""

    def test_is_sequenced_and_short(self):
        tutorial = tutorials.get('lo')
        self.assertEqual(tutorial.order, 'sequence')
        self.assertLessEqual(len(tutorial.steps), 8,
                             'lo has grown back into a grab-bag')

    def test_reaches_launch(self):
        """The point of `lo` is that it gets you to an event sample."""

        keys = [step.key for step in tutorials.get('lo').steps]
        for expected in ('generate', 'output', 'launch'):
            self.assertIn(expected, keys,
                          'lo no longer walks the user through %r' % expected)

    def test_intro_adapts_to_madspace(self):
        """Step 0 asks for `install madspace` only when it is missing."""

        import madgraph.interface.tutorials.lo as lo_module

        original = lo_module.madspace_is_installed
        try:
            lo_module.madspace_is_installed = lambda: False
            self.assertIn('install madspace', lo_module.intro(None))
            lo_module.madspace_is_installed = lambda: True
            self.assertNotIn('install madspace', lo_module.intro(None))
        finally:
            lo_module.madspace_is_installed = original

    def test_see_also_never_points_at_a_missing_tutorial(self):
        """The closing signposts list only tutorials that actually exist."""

        text = tutorials.get('lo').steps[-1].render(None)
        registered = set(tutorials.names(include_aliases=True))
        # 'stop', 'list' and 'status' are sub-commands, not tutorials
        registered |= {'stop', 'list', 'status'}
        for name in re.findall(r'`tutorial ([a-z0-9_]+)`', text):
            self.assertIn(name, registered,
                          'lo points at `tutorial %s`, which does not exist'
                          % name)


#===============================================================================
# the registry
#===============================================================================

class TestRegistry(unittest.TestCase):

    def test_primary_names(self):
        names = [t.name for t in tutorials.all_tutorials()]
        for expected in ('lo', 'nlo', 'madloop'):
            self.assertIn(expected, names)

    def test_legacy_names_are_aliases(self):
        """The pre-2026 names keep working, and resolve to the new ones."""

        for old, new in (('MadGraph5', 'lo'), ('aMCatNLO', 'nlo'),
                         ('MadLoop', 'madloop')):
            self.assertIsNotNone(tutorials.get(old), '%s no longer accepted' % old)
            self.assertEqual(tutorials.get(old).name, new)

    def test_lookup_is_case_insensitive(self):
        self.assertEqual(tutorials.get('MADGRAPH5').name, 'lo')
        self.assertEqual(tutorials.get('LO').name, 'lo')

    def test_unknown_name(self):
        self.assertIsNone(tutorials.get('not-a-tutorial'))

    def test_every_step_has_a_title(self):
        for tutorial in tutorials.all_tutorials(include_hidden=True):
            for step in tutorial.steps:
                self.assertTrue(step.title,
                                'step %r of %s has no title' % (step.key, tutorial.name))


#===============================================================================
# the session
#===============================================================================

class TestKeys(unittest.TestCase):
    """The command line -> key reduction must match the old getattr chain."""

    def test_two_word_key_first(self):
        self.assertEqual(TutorialSession.keys_for('import model sm'),
                         ('import_model', 'import'))

    def test_dot_is_truncated(self):
        self.assertEqual(TutorialSession.keys_for('open index.html'),
                         ('open_index', 'open'))

    def test_single_word(self):
        self.assertEqual(TutorialSession.keys_for('customize_model'),
                         ('customize_model',))

    def test_empty(self):
        self.assertEqual(TutorialSession.keys_for('   '), ())


class TestOrdering(_TutorialTestCase):

    def _tutorial(self, order):
        return Tutorial(
            name='fake-%s' % order, title='fake', order=order,
            steps=[Step('generate', 'FIRST', title='one'),
                   Step('output', 'SECOND', title='two'),
                   Step('generate', 'THIRD', title='three')])

    def test_sequence_advances_through_repeated_commands(self):
        """A tutorial teaching `generate` twice gets two different lessons.

        This is what the old command-name lookup could not express and is the
        reason the engine exists.
        """

        session = TutorialSession(self._tutorial('sequence'))
        self.assertEqual(session.step_for('generate p p > t t~')[1].text, 'FIRST')
        session.advance(0)
        self.assertEqual(session.step_for('output PROC')[1].text, 'SECOND')
        session.advance(1)
        self.assertEqual(session.step_for('generate p p > z')[1].text, 'THIRD')

    def test_sequence_does_not_go_backwards(self):
        session = TutorialSession(self._tutorial('sequence'))
        session.advance(0)
        # 'generate' matches step 0 and step 2; already past 0, so it is step 2
        self.assertEqual(session.step_for('generate p p > z')[1].text, 'THIRD')

    def test_sequence_does_not_refire_the_current_step(self):
        session = TutorialSession(self._tutorial('sequence'))
        session.advance(2)
        self.assertIsNone(session.step_for('generate p p > z'))

    def test_free_always_takes_the_first_match(self):
        session = TutorialSession(self._tutorial('free'))
        session.advance(2)
        self.assertEqual(session.step_for('generate p p > z')[1].text, 'FIRST')

    def test_specific_key_wins_over_bare_one(self):
        """'open index.html' must resolve to the open_index step even though a
        plain 'open' step sits earlier in the list."""

        tutorial = Tutorial(name='fake-keys', title='fake', order='free',
                            steps=[Step('open', 'BARE', title='bare'),
                                   Step('open_index', 'SPECIFIC', title='specific')])
        session = TutorialSession(tutorial)
        self.assertEqual(session.step_for('open index.html')[1].text, 'SPECIFIC')
        self.assertEqual(session.step_for('open other.txt')[1].text, 'BARE')

    def test_no_match(self):
        session = TutorialSession(self._tutorial('free'))
        self.assertIsNone(session.step_for('display particles'))


#===============================================================================
# attach / detach
#===============================================================================

class TestAttach(_TutorialTestCase):

    def test_attach_and_detach_restore_the_class(self):
        interface = _BareInterface()
        original = interface.__class__
        tutorial_mixin.attach(interface, tutorials.start('lo'))
        self.assertTrue(tutorial_mixin.is_attached(interface))
        self.assertIsInstance(interface, original)
        tutorial_mixin.detach(interface)
        self.assertFalse(tutorial_mixin.is_attached(interface))
        self.assertIs(interface.__class__, original)

    def test_no_output_once_detached(self):
        interface = _BareInterface()
        tutorial_mixin.attach(interface, tutorials.start('lo'))
        tutorial_mixin.detach(interface)
        self.capture.messages = []
        interface.postcmd(True, 'generate p p > t t~')
        self.assertEqual(self.capture.messages, [])

    def test_attaching_twice_keeps_one_layer(self):
        interface = _BareInterface()
        original = interface.__class__
        tutorial_mixin.attach(interface, tutorials.start('lo'))
        wrapped = interface.__class__
        tutorial_mixin.attach(interface, tutorials.start('nlo'))
        self.assertIs(interface.__class__, wrapped)
        tutorial_mixin.detach(interface)
        self.assertIs(interface.__class__, original)

    def test_switching_tutorial_switches_content(self):
        interface = _BareInterface()
        tutorial_mixin.attach(interface, tutorials.start('lo'))
        tutorial_mixin.attach(interface, tutorials.start('madloop'))
        self.capture.messages = []
        interface.postcmd(True, 'check')
        self.assertEqual(self.capture.messages,
                         [retarget(legacy_madloop.check).replace('\n', '\n\t')])


#===============================================================================
# every tutorial is walkable
#===============================================================================

class TestTutorialsAreWalkable(_TutorialTestCase):
    """A tutorial has to be followable by doing what it says.

    Each step names, in `solution`, the command it asks the user to type; the
    step that command triggers is the next one.  So walking a sequenced
    tutorial by typing its own solutions must visit every step in order.  If a
    lesson ever asks for a command that does not lead anywhere, this fails.
    """

    def test_syntax_solutions_lead_to_the_next_step(self):
        tutorial = tutorials.get('syntax')
        self.assertIsNotNone(tutorial, 'the syntax tutorial is not registered')
        self.assertEqual(tutorial.order, 'sequence')

        session = TutorialSession(tutorial)
        for index, step in enumerate(tutorial.steps[:-1]):
            self.assertTrue(step.solution,
                            'step %d (%s) asks for no command' % (index, step.title))
            session.index = index
            found = session.step_for(step.solution)
            self.assertIsNotNone(
                found, 'step %d (%s) asks for %r, which triggers nothing'
                % (index, step.title, step.solution))
            self.assertEqual(
                found[0], index + 1,
                'step %d (%s) asks for %r, which jumps to step %d rather than %d'
                % (index, step.title, step.solution, found[0], index + 1))

    def test_every_sequenced_step_makes_progress(self):
        """Weaker rule, applied to every sequenced tutorial: doing what a step
        asks must move you forward.  `lo` deliberately skips a step -- the
        intro jumps past `install madspace` when madspace is already there --
        so it cannot use the strict rule above."""

        for tutorial in tutorials.all_tutorials(include_hidden=True):
            if tutorial.order != 'sequence':
                continue
            session = TutorialSession(tutorial)
            for index, step in enumerate(tutorial.steps[:-1]):
                self.assertTrue(
                    step.solution,
                    '%s step %d (%s) asks for no command'
                    % (tutorial.name, index, step.title))
                session.index = index
                found = session.step_for(step.solution)
                self.assertIsNotNone(
                    found, '%s step %d (%s) asks for %r, which triggers nothing'
                    % (tutorial.name, index, step.title, step.solution))
                self.assertGreater(
                    found[0], index,
                    '%s step %d (%s) asks for %r, which does not move forward'
                    % (tutorial.name, index, step.title, step.solution))

    def test_sequenced_tutorials_start_on_the_tutorial_command(self):
        """The first step is the intro, triggered by `tutorial NAME` itself."""

        for tutorial in tutorials.all_tutorials(include_hidden=True):
            first = tutorial.steps[0]
            self.assertTrue(first.matches(('tutorial',)),
                            'tutorial %s does not open with an intro step'
                            % tutorial.name)

    def test_last_step_is_terminal(self):
        for tutorial in tutorials.all_tutorials(include_hidden=True):
            session = TutorialSession(tutorial)
            session.index = len(tutorial.steps) - 1
            self.assertTrue(session.finished)
            self.assertIsNone(session.next_step)


#===============================================================================
# the `tutorial` command itself
#===============================================================================

class _BareMadGraphCmd(mg_interface.MadGraphCmd):
    """A MadGraphCmd without the real (expensive) __init__.

    Carries only what check_tutorial / ask_tutorial touch -- and deliberately
    *not* `force`, which is set opportunistically elsewhere (by
    `import command -f`) and is not an attribute every interface has.  Assuming
    it existed crashed a bare `tutorial` on the real MasterCmd.
    """

    def __init__(self, use_rawinput=False):
        self.use_rawinput = use_rawinput
        self.asked = None

    def ask(self, question, default, choices=(), **opts):
        self.asked = (question, default, list(choices))
        return self._answer


class TestTutorialCommand(unittest.TestCase):

    def test_bare_tutorial_does_not_need_a_force_attribute(self):
        """The regression: `tutorial` with no argument on a fresh interface."""

        interface = _BareMadGraphCmd()
        self.assertFalse(hasattr(interface, 'force'))
        args = []
        interface.check_tutorial(args)          # must not raise
        self.assertEqual(args, [tutorials.all_tutorials()[0].name])

    def test_bare_tutorial_never_blocks_without_a_terminal(self):
        """In a command file or a pipe it keeps its historical meaning."""

        interface = _BareMadGraphCmd(use_rawinput=False)
        interface._answer = 'should not be asked'
        self.assertEqual(interface.ask_tutorial(), 'lo')
        self.assertIsNone(interface.asked, 'the menu blocked on a non-tty')

    def test_force_skips_the_menu(self):
        interface = _BareMadGraphCmd(use_rawinput=True)
        interface.force = True
        interface._answer = 'should not be asked'
        self.assertEqual(interface.ask_tutorial(), 'lo')
        self.assertIsNone(interface.asked)

    def test_a_number_picks_that_tutorial(self):
        for index, tutorial in enumerate(tutorials.all_tutorials()):
            interface = _BareMadGraphCmd()
            interface._answer = str(index + 1)
            self.assertEqual(self._ask_interactively(interface), tutorial.name)

    def _ask_interactively(self, interface):
        """Run ask_tutorial as though stdin were a terminal."""

        import sys as _sys

        class _Tty(object):
            def __getattr__(self, name):
                return getattr(_sys.__stdin__, name)

            def isatty(self):
                return True

        interface.use_rawinput = True
        saved = _sys.stdin
        _sys.stdin = _Tty()
        try:
            return interface.ask_tutorial()
        finally:
            _sys.stdin = saved

    def test_a_name_is_accepted_too(self):
        interface = _BareMadGraphCmd()
        interface._answer = 'syntax'
        self.assertEqual(self._ask_interactively(interface), 'syntax')

    def test_the_menu_offers_names_numbers_and_stop(self):
        interface = _BareMadGraphCmd()
        interface._answer = 'lo'
        self._ask_interactively(interface)
        _question, default, choices = interface.asked
        self.assertEqual(default, 'lo')
        for name in tutorials.names():
            self.assertIn(name, choices)
        self.assertIn('1', choices)
        self.assertIn('stop', choices)

    def test_aliases_are_normalised_to_the_primary_name(self):
        for old, new in (('MadGraph5', 'lo'), ('aMCatNLO', 'nlo'),
                         ('MadLoop', 'madloop')):
            args = [old]
            _BareMadGraphCmd().check_tutorial(args)
            self.assertEqual(args, [new])

    def test_subcommands_pass_through(self):
        for name in ('stop', 'list', 'status'):
            args = [name]
            _BareMadGraphCmd().check_tutorial(args)
            self.assertEqual(args, [name])

    def test_an_unknown_name_is_refused(self):
        interface = _BareMadGraphCmd()
        self.assertRaises(madgraph.InvalidCmd,
                          interface.check_tutorial, ['not-a-tutorial'])

    def test_too_many_arguments_is_refused(self):
        interface = _BareMadGraphCmd()
        self.assertRaises(madgraph.InvalidCmd,
                          interface.check_tutorial, ['lo', 'syntax'])
