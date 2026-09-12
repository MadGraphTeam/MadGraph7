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
"""Data model for the tutorial mode: Step, Tutorial and TutorialSession.

A tutorial is an ordered list of steps.  A step is triggered by a command the
user types; when it triggers, its text is printed and it becomes the current
step.  Two matching policies exist:

  'free'      any step whose key matches fires, wherever it sits in the list.
              This is the behaviour the pre-2026 tutorials had (they were a
              plain command-name -> text lookup) and the ported ones keep it.
  'sequence'  only the current step, or a later one, may fire.  A tutorial that
              teaches the same command several times over (`generate` five
              times with a different lesson each time) needs this.

Nothing here talks to the interface; the mixin does that.
"""

from __future__ import absolute_import

import os


class Step(object):
    """One lesson of a tutorial.

    key       command key that triggers the step.  A command line is reduced to
              candidate keys by TutorialSession.keys_for(): 'import model sm'
              gives ('import_model', 'import'), 'generate p p > t t~' gives
              ('generate',).  A step matches if its key is any of them; a
              callable key is called with (keys, line, interface) instead.
    text      printed when the step fires.  Either a string, or a
              callable(interface) -> str for a lesson whose wording depends on
              the machine it runs on (whether madspace is already installed,
              say).
    hint      printed by the `hint` command; falls back to nothing.
    solution  the command line this step is waiting for.  Printed by `next` and
              `solution` -- never executed, the user always types it.  May be a
              callable(interface) -> str, for a step whose command depends on
              what the user chose earlier (the name they gave `output`, say);
              resolve it with get_solution().
    requires  list of prerequisite names checked before the step is announced.
    setup     callable(interface) run before the step is announced.
    question_hint
              shown under any question MG7 asks while this step is current, in
              place of the generic "type 'help'" line. The card question a
              `launch` step leads to is the case that matters: it is asked by
              the run interface in the middle of the command, so this is the
              only way a step can say anything there.
    """

    def __init__(self, key, text, hint=None, solution=None, requires=None,
                 setup=None, title=None, question_hint=None):
        self.key = key
        self.text = text
        self.hint = hint
        self.solution = solution
        self.requires = list(requires) if requires else []
        self.setup = setup
        self.title = title
        self.question_hint = question_hint

    def render(self, interface=None):
        """The text to print for this step."""

        if callable(self.text):
            return self.text(interface)
        return self.text

    def get_solution(self, interface=None):
        """The command this step is waiting for, resolved against the session.

        Falls back to a callable's answer for a missing interface, so `hint`
        and the tests both work before anything has been generated.
        """

        if callable(self.solution):
            try:
                return self.solution(interface)
            except Exception:
                return None
        return self.solution

    def matches(self, keys, line=None, interface=None):
        """True if this step is triggered by a command reduced to `keys`."""

        if callable(self.key):
            return bool(self.key(keys, line, interface))
        if isinstance(self.key, (list, tuple, set)):
            return any(k in keys for k in self.key)
        return self.key in keys

    def __repr__(self):
        return '<Step %s>' % (self.title or self.key)


class Exercise(Step):
    """A step that asks the user to do something and checks whether they did.

    question  what the user is asked to do.  Printed when the exercise becomes
              current, which is when the *previous* step is completed.
    check     callable(interface, line) -> True to pass, False to fail, or a
              string to fail with that message.  It must inspect the state the
              command produced -- interface._curr_amps and friends -- never the
              text of the line: `p p > t t~ QED=0` and `p p > t t~ QCD=2 QED=0`
              are both right, and string matching would fail whoever typed the
              second one.
    mistakes  [(predicate(interface, line), explanation)], tried in order when
              the check fails.  These are the wrong answers worth naming; each
              one is a teaching moment.
    report    optional callable(interface) -> str appended to the pass message,
              for saying what the user's command actually produced.

    A failed attempt never blocks and never advances: the command has already
    run, the user sees what it did, and they can try again, ask for a `hint`,
    or `skip`.
    """

    def __init__(self, key, question, check, mistakes=(), hint=None,
                 solution=None, title=None, praise=None, report=None,
                 question_hint=None):
        Step.__init__(self, key, question, hint=hint, solution=solution,
                      title=title, question_hint=question_hint)
        self.question = question
        self.check = check
        self.mistakes = list(mistakes)
        self.praise = praise
        self.report = report

    def evaluate(self, interface, line):
        """Mark an attempt.  Returns (passed, message)."""

        try:
            verdict = self.check(interface, line)
        except Exception as error:
            return False, ("That command did not leave anything to check "
                           "(%s). Try again, or `skip`." % error)

        if verdict is True:
            message = self.praise or 'That is it.'
            if self.report:
                try:
                    message = '%s\n\n%s' % (message, self.report(interface))
                except Exception:
                    pass
            return True, message

        if isinstance(verdict, str):
            return False, verdict

        for predicate, explanation in self.mistakes:
            try:
                if predicate(interface, line):
                    return False, explanation
            except Exception:
                continue
        return False, self.fallback(interface, line)

    def fallback(self, interface, line):
        """What an answer we did not anticipate gets told.

        This has to be good: it is the only thing standing between an unusual
        wrong answer and a bare "no".
        """

        return ('Not quite -- and not a mistake this exercise knows by name.\n'
                '\nWhat was asked:\n  %s\n'
                '\nWhat your command produced:\n%s\n'
                '\nTry again, or `hint`, or `solution` to see one right '
                'answer, or `skip`.'
                % (self.solution or self.title or 'see the question above',
                   describe_state(interface)))


def describe_state(interface, indent='  '):
    """A short readable summary of what the interface currently holds.

    Used by Exercise.fallback, so an unanticipated answer still gets told what
    it actually did rather than just that it was wrong.
    """

    amps = getattr(interface, '_curr_amps', None)
    if not amps:
        return indent + 'no process is defined'

    lines = []
    for amp in amps[:4]:
        try:
            process = core_process(amp)
            bits = [process.base_string()]
            orders = dict(process.get('orders'))
            squared = dict(process.get('squared_orders'))
            if orders:
                bits.append('orders %s' % _fmt_orders(orders))
            if squared:
                types = dict(process.get('sqorders_types'))
                bits.append('squared orders %s'
                            % _fmt_orders(squared, types))
            bits.append('%d diagrams' % amp.get_number_of_diagrams())
            lines.append(indent + ', '.join(bits))
        except Exception:
            continue
    if len(amps) > 4:
        lines.append(indent + '... and %d more' % (len(amps) - 4))
    return '\n'.join(lines) or (indent + 'nothing that could be summarised')


def lhapdf_configured(interface):
    """Whether this MG7 has a usable lhapdf-config.

    Scale and PDF variations are computed by systematics.py, which imports the
    python lhapdf module and finds it through lhapdf-config
    (mg7/launch.py:_lhapdf_config_path).  Without it the run still produces a
    cross section and its integration error -- only the theory uncertainty is
    missing.  Reported as a tri-state so a tutorial can say "not configured
    here" when it knows, and stay general when it cannot tell.
    """

    try:
        import madgraph.various.misc as misc
    except Exception:
        return None

    try:
        configured = interface.options.get('lhapdf')
    except Exception:
        configured = None

    for candidate in (configured, 'lhapdf-config'):
        if not candidate:
            continue
        if os.path.isabs(candidate):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return True
        elif misc.which(candidate):
            return True
    return False


def pythia8_available(interface):
    """Whether this MG7 has a usable Pythia8 *and* the MG5aMC interface to it.

    MG5 nulls options['pythia8_path'] at startup when the headers are not where
    it points (madgraph_interface.py:7912), and showering from aMC@NLO also
    needs options['mg5amc_py8_interface_path'], so both are checked.
    """

    try:
        options = interface.options
    except Exception:
        return False
    for key in ('pythia8_path', 'mg5amc_py8_interface_path'):
        value = options.get(key)
        if value in (None, '', 'None'):
            return False
    return True


def output_name(interface, default):
    """The directory the user's last `output` actually made.

    A tutorial says "you now have a directory called X", and X has to be the
    name the user chose -- they are not obliged to use the one we suggested.
    `_done_export` is [path, mode]; falls back to the suggested name before any
    output exists.
    """

    try:
        done = getattr(interface, '_done_export', None)
        if done:
            name = os.path.basename(os.path.normpath(done[0]))
            if name:
                return name
    except Exception:
        pass
    return default


def total_diagrams(interface):
    """How many diagrams the current process(es) came to, over all of them."""

    total = 0
    for amplitude in getattr(interface, '_curr_amps', None) or []:
        try:
            total += amplitude.get_number_of_diagrams()
        except Exception:
            continue
    return total


def applied_orders(interface):
    """The coupling orders MG5 actually put on the current process(es).

    Returns a list of the distinct order dicts, so a tutorial can describe what
    really happened rather than assert what usually happens.  MG5 has more than
    one way of settling this -- a minimal-WEIGHTED search, or an explicit order
    -- and which one you get depends on the process.
    """

    seen = []
    for amplitude in getattr(interface, '_curr_amps', None) or []:
        try:
            orders = dict(core_process(amplitude).get('orders'))
        except Exception:
            continue
        if orders not in seen:
            seen.append(orders)
    return seen


def describe_applied_orders(interface, indent=''):
    """Prose for what MG5 chose, matched to what it actually chose."""

    found = [orders for orders in applied_orders(interface) if orders]
    if not found:
        return (indent + 'MG5 put no coupling-order constraint on the process '
                'at all, so every diagram the model allows is included.')

    # an amplitude order means '<=' -- MG5 warns "Interpreting 'QED=2' as
    # 'QED<=2'" and prints WEIGHTED<=2 -- so quote it the way the user just
    # saw it
    shown = ' and '.join(_fmt_orders(orders, default='<=') or 'no constraint'
                         for orders in found[:3])

    if any('WEIGHTED' in orders for orders in found):
        return (indent + 'MG5 settled on **%s**.\n\n'
                % shown +
                indent + 'WEIGHTED counts QCD + 2*QED, and with no orders given '
                'MG5 searches:\nit takes the lowest WEIGHTED that produces any '
                'diagram at all. Here that\nlands on the QCD diagrams. It is a '
                'search, not a statement of physics.')
    return (indent + 'MG5 settled on **%s**.\n\n' % shown +
            indent + 'With no orders given MG5 works out a constraint for you '
            'and applies it\ndirectly. It is a choice made on your behalf, not '
            'a statement of physics.')


def _fmt_orders(orders, types=None, default='='):
    types = types or {}
    return ' '.join('%s%s%s' % (name, types.get(name, default), value)
                    for name, value in sorted(orders.items()))


def core_process(amplitude):
    """The production process of an amplitude, decay chain or not.

    A DecayChainAmplitude has no 'process' of its own -- asking for one raises
    KeyError -- so the core process has to be dug out of its first amplitude.
    """

    try:
        return amplitude.get('process')
    except Exception:
        return amplitude.get('amplitudes')[0].get('process')


class Tutorial(object):
    """An ordered list of steps, addressable by name.

    name        primary name, what `tutorial NAME` takes and what the menu shows.
    title       one line for the menu.
    description longer blurb for `tutorial list`.
    steps       list of Step.
    aliases     other accepted names; hidden from the menu.  Used to keep the
                pre-2026 names (MadGraph5 / aMCatNLO / MadLoop) working.
    order       'free' or 'sequence', see the module docstring.
    requires    prerequisites checked when the tutorial starts.
    hidden      keep out of the menu (for tutorials still being written).
    see_also    names of related tutorials, rendered as a "where to go next"
                block.  Names that are not registered are dropped, so a
                tutorial can point at one that has not been written yet
                without ever advertising a dead end.
    section     which group of the menu this belongs to: 'basic', 'advanced'
                or 'exercises'.
    ai_generated
                True when the content was written by an AI and has not been
                validated by the developers.  The menu says so, per section,
                and this is what keeps that notice honest for the tutorials
                carried over from the pre-2026 hand-written text.
    """

    SECTIONS = ('basic', 'advanced', 'exercises')

    def __init__(self, name, title, steps, description='', aliases=(),
                 order='free', requires=None, hidden=False, see_also=(),
                 section='advanced', ai_generated=True):
        self.name = name
        self.title = title
        self.description = description or title
        self.steps = list(steps)
        self.aliases = tuple(aliases)
        if order not in ('free', 'sequence'):
            raise ValueError("unknown step order %r" % order)
        self.order = order
        self.requires = list(requires) if requires else []
        self.hidden = hidden
        self.see_also = tuple(see_also)
        if section not in self.SECTIONS:
            raise ValueError('unknown tutorial section %r' % section)
        self.section = section
        self.ai_generated = bool(ai_generated)

    @property
    def names(self):
        return (self.name,) + self.aliases

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        return '<Tutorial %s (%d steps)>' % (self.name, len(self.steps))


class TutorialSession(object):
    """A tutorial plus where the user has got to in it."""

    def __init__(self, tutorial):
        self.tutorial = tutorial
        self.index = -1          # -1: not started; else index of the last step fired
        self.seen = set()
        # set by do_tutorial for informational sub-commands ('list', 'status')
        # so that they do not re-trigger the intro step
        self.suppress_next = False

    # -- command line -> candidate keys ---------------------------------------

    @staticmethod
    def keys_for(line):
        """Reduce a command line to the keys a step may match.

        Mirrors exactly what madgraph_interface.CmdExtended.postcmd used to do:
        the two-word key first ('import model sm' -> 'import_model', with the
        second word truncated at a dot so 'open index.html' -> 'open_index'),
        then the bare command.  Order matters: the more specific key wins.
        """

        args = line.split()
        if not args:
            return ()
        if len(args) == 1:
            return (args[0],)
        return (args[0] + '_' + args[1].split('.')[0], args[0])

    # -- advancing -------------------------------------------------------------

    def step_for(self, line, interface=None):
        """Return (index, step) for the step triggered by `line`, or None.

        Candidate keys are tried most-specific first, and for each key every
        allowed step is scanned.  That ordering -- key-major, not step-major --
        is what reproduces the old getattr() chain exactly: 'open index.html'
        resolves to the 'open_index' step even if a plain 'open' step sits
        earlier in the list.  Steps with a callable key are tried last, since
        they cannot be indexed by key.

        The session is left untouched; advance() commits.
        """

        keys = self.keys_for(line)
        if not keys:
            return None

        steps = self.tutorial.steps
        if self.tutorial.order == 'free':
            allowed = list(range(len(steps)))
        else:
            # strictly *after* the current step: a sequenced tutorial teaches
            # the same command several times over, so re-matching the step we
            # just fired would pin the user on lesson one forever
            allowed = list(range(self.index + 1, len(steps)))

        for key in keys:
            for i in allowed:
                if callable(steps[i].key):
                    continue
                if steps[i].matches((key,), line, interface):
                    return i, steps[i]

        for i in allowed:
            if callable(steps[i].key) and steps[i].matches(keys, line, interface):
                return i, steps[i]
        return None

    def advance(self, index):
        self.index = index
        self.seen.add(index)

    @property
    def current(self):
        if 0 <= self.index < len(self.tutorial.steps):
            return self.tutorial.steps[self.index]
        return None

    @property
    def next_step(self):
        if self.index + 1 < len(self.tutorial.steps):
            return self.tutorial.steps[self.index + 1]
        return None

    @property
    def finished(self):
        return self.index >= len(self.tutorial.steps) - 1

    def question_hint(self):
        """The hint for whatever step is current, or None.

        The step that *asked* for the command is the current one while that
        command runs, which is what makes this reach the card question a
        `launch` step leads to.
        """

        step = self.current
        return step.question_hint if step is not None else None

    def progress(self):
        """(done, total) for the prompt and `status`."""
        return (max(self.index, 0) + (1 if self.index >= 0 else 0),
                len(self.tutorial.steps))
