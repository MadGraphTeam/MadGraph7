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
"""The interface layer of the tutorial mode.

While a tutorial runs, `TutorialMixin` is spliced in front of the *instance's*
own class -- attach() rebinds interface.__class__ -- rather than in front of
the class the switcher dispatches to.  That is deliberate and it is the only
thing that works: cmd dispatch resolves `do_XXX`, `postcmd` and `default` on
the instance (extended_cmd.Cmd.onecmd_orig does `getattr(self, 'do_' + cmd)`),
while Switcher.self.cmd is only consulted for the commands Switcher explicitly
forwards.  Wrapping self.cmd would therefore reach neither postcmd nor any
command the mixin adds.

Splicing at the instance also makes the tutorial automatically survive
Switcher.change_principal_cmd: the LO <-> NLO switch swaps self.cmd, not the
instance class, so a tutorial that crosses `generate p p > t t~ [QCD]` keeps
running with no extra hook.

The mixin never blocks a command: a wrong or off-script command runs exactly as
it would outside tutorial mode, and the tutorial only comments on it.
"""

from __future__ import absolute_import

import logging

import madgraph.interface.extended_cmd as extended_cmd
import madgraph.various.misc as misc
from madgraph.interface.tutorials._style import to_terminal
from madgraph.interface.tutorials.session import Exercise

logger_tuto = logging.getLogger('tutorial')
logger = logging.getLogger('madgraph')


def emit(text):
    """Print one tutorial block, in the format the tutorial logger frames."""

    logger_tuto.info(to_terminal(text).replace('\n', '\n\t'))


class TutorialMixin(object):
    """Interface overrides active only while a tutorial is running."""

    # the session is stored on the interface instance as _tutorial_session
    # and the pre-tutorial class as _tutorial_base_class

    # -- the hook that advances the tutorial ----------------------------------

    def postcmd(self, stop, line):
        stop = super(TutorialMixin, self).postcmd(stop, line)
        if stop is False:
            return False

        session = getattr(self, '_tutorial_session', None)
        if session is None:
            return stop

        # Only react to what the user actually typed.  MG5 runs plenty of
        # commands for itself -- importing a model issues half a dozen 'define'
        # commands, `display diagrams` issues an `open` -- and those used to
        # fire tutorial steps, printing the same block six times over and, in a
        # sequenced tutorial, skipping the user several lessons ahead.
        # exec_cmd tracks the nesting depth and a user command sits at 0, both
        # interactively and from a command file; anything deeper is MG5 talking
        # to itself.
        if getattr(self, 'exec_cmd_depth', 0) > 0:
            return stop

        if session.suppress_next:
            session.suppress_next = False
            return stop

        found = session.step_for(line, self)
        if found is None:
            return stop

        index, step = found
        if step.setup:
            step.setup(self)

        if isinstance(step, Exercise):
            passed, message = step.evaluate(self, line)
            if not passed:
                # never blocks and never advances: the command has already run,
                # so the user can simply try again
                emit(message)
                return stop
            session.advance(index)
            emit(self._tutorial_join(message, session))
            return stop

        session.advance(index)
        emit(step.render(self))
        return stop

    def _tutorial_join(self, message, session):
        """A passed exercise's verdict, followed by whatever comes next.

        An exercise is triggered by the user's answer, so the *next* question
        has to be printed here rather than waiting for a command that would
        trigger it.
        """

        following = session.next_step
        if following is None:
            return '%s\n\nThat was the last one.' % message
        if isinstance(following, Exercise):
            return '%s\n\n%s' % (message, following.question)
        # a plain step after the exercises: the closing text.  Nothing will
        # ever trigger it, so show it now and mark the tutorial finished.
        session.advance(session.index + 1)
        return '%s\n%s' % (message, following.render(self))

    # -- tutorial-only commands ----------------------------------------------
    #
    # None of these execute anything: `next` and `solution` print the command
    # and the user types it.

    def do_hint(self, line):
        """Not in help: show a hint for the current tutorial step"""
        step = self._tutorial_step_or_warn()
        if step is None:
            return
        solution = step.get_solution(self)
        if step.hint:
            emit(step.hint)
        elif solution:
            emit("Try:\n%s%s" % (self._tutorial_prompt_text(), solution))
        else:
            emit("No hint for this step -- try 'solution'.")

    def do_solution(self, line):
        """Not in help: show the command the current tutorial step expects"""
        self._tutorial_show_solution(self._tutorial_step_or_warn())

    def do_next(self, line):
        """Not in help: show the next command of the tutorial"""
        session = self._tutorial_session_or_warn()
        if session is None:
            return
        step = session.current or session.next_step
        self._tutorial_show_solution(step)

    def do_repeat(self, line):
        """Not in help: print the current tutorial step again"""
        step = self._tutorial_step_or_warn()
        if step is not None:
            emit(step.render(self))

    def do_back(self, line):
        """Not in help: go back one tutorial step"""
        session = self._tutorial_session_or_warn()
        if session is None:
            return
        if session.index <= 0:
            emit("You are at the first step of this tutorial.")
            return
        session.index -= 1
        emit(session.current.render(self))

    def do_skip(self, line):
        """Not in help: skip the current tutorial step"""
        session = self._tutorial_session_or_warn()
        if session is None:
            return
        step = session.next_step
        if step is None:
            emit("That was the last step of this tutorial.\n"
                 "Type 'tutorial' to pick another one, or 'tutorial stop'.")
            return
        session.advance(session.index + 1)
        emit(step.render(self))

    # -- helpers ---------------------------------------------------------------

    def _tutorial_session_or_warn(self):
        session = getattr(self, '_tutorial_session', None)
        if session is None:
            logger.warning('No tutorial is running.')
        return session

    def _tutorial_step_or_warn(self):
        session = self._tutorial_session_or_warn()
        if session is None:
            return None
        step = session.current
        if step is None:
            step = session.next_step
        return step

    def _tutorial_show_solution(self, step):
        if step is None:
            return
        solution = step.get_solution(self)
        if not solution:
            emit("This step has no single command to give -- read it again "
                 "with 'repeat'.")
            return
        emit("The tutorial expects:\n%s%s"
             % (self._tutorial_prompt_text(), solution))

    @staticmethod
    def _tutorial_prompt_text():
        """The bare prompt, for quoting commands in tutorial text."""
        import madgraph.interface.madgraph_interface as mg
        return mg.MG7_PROMPT_TEXT


#===============================================================================
# splicing the mixin in and out of a live interface
#===============================================================================

_WRAPPED = {}       # base class -> class with the mixin in front


def _wrap(base):
    if base not in _WRAPPED:
        _WRAPPED[base] = type('%sTutorial' % base.__name__,
                              (TutorialMixin, base), {})
    return _WRAPPED[base]


def is_attached(interface):
    return isinstance(interface, TutorialMixin)


def attach(interface, session):
    """Start `session` on `interface`, splicing the mixin in if needed."""

    if not is_attached(interface):
        interface._tutorial_base_class = interface.__class__
        interface.__class__ = _wrap(interface.__class__)
        _suspend_crash_on_error(interface)
    interface._tutorial_session = session
    _arm_question_hooks(session)
    return session


def detach(interface):
    """Stop any running tutorial and restore the plain interface class."""

    session = getattr(interface, '_tutorial_session', None)
    interface._tutorial_session = None
    if is_attached(interface):
        base = getattr(interface, '_tutorial_base_class', None)
        if base is not None:
            interface.__class__ = base
        interface._tutorial_base_class = None
        _restore_crash_on_error(interface)
    _disarm_question_hooks()
    return session


def _suspend_crash_on_error(interface):
    """Turn crash_on_error off for the life of the tutorial.

    A tutorial is a place to make mistakes -- that is what the exercises are
    for -- and with crash_on_error set, a mistyped command does not just fail,
    it tears down the whole session. Suspend it while a tutorial runs and put
    the user's setting back on `tutorial stop`.

    TMP_variable installs the new value on construction and restores it in
    __exit__, so it can span the tutorial rather than a single block; it
    addresses `options` by key, leaving any other option the user changes
    meanwhile alone.
    """

    interface._tutorial_crash_guard = None
    options = getattr(interface, 'options', None)
    if not isinstance(options, dict) or 'crash_on_error' not in options:
        return
    if not options['crash_on_error']:
        return          # already off; nothing to suspend or restore
    interface._tutorial_crash_guard = misc.TMP_variable(
        options, 'crash_on_error', False)


def _restore_crash_on_error(interface):
    guard = getattr(interface, '_tutorial_crash_guard', None)
    if guard is not None:
        guard.__exit__(None, None, None)
    interface._tutorial_crash_guard = None


def mixin_command_names():
    """`do_XXX` names the mixin adds -- Switcher.debug_link_to_command skips
    these, since they are deliberately not forwarded through self.cmd."""

    return [name for name in vars(TutorialMixin) if name.startswith('do_')]


def _arm_question_hooks(session):
    """Let the running tutorial speak at any question, and stop the clock.

    Both are module-level switches in extended_cmd because a question is often
    asked by an object the mixin is not attached to -- the launch card question
    belongs to the run interface, not to the command the user typed `launch`
    at.
    """

    extended_cmd.question_hint = lambda: _question_hint(session)
    extended_cmd.suppress_timeout = True


def _disarm_question_hooks():
    extended_cmd.question_hint = None
    extended_cmd.suppress_timeout = False


def _question_hint(session):
    """The current step's hint, styled, or None to keep the generic line."""

    hint = session.question_hint()
    return to_terminal(hint) if hint else None
