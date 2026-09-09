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
              `solution` -- never executed, the user always types it.
    requires  list of prerequisite names checked before the step is announced.
    setup     callable(interface) run before the step is announced.
    """

    def __init__(self, key, text, hint=None, solution=None, requires=None,
                 setup=None, title=None):
        self.key = key
        self.text = text
        self.hint = hint
        self.solution = solution
        self.requires = list(requires) if requires else []
        self.setup = setup
        self.title = title

    def render(self, interface=None):
        """The text to print for this step."""

        if callable(self.text):
            return self.text(interface)
        return self.text

    def matches(self, keys, line=None, interface=None):
        """True if this step is triggered by a command reduced to `keys`."""

        if callable(self.key):
            return bool(self.key(keys, line, interface))
        if isinstance(self.key, (list, tuple, set)):
            return any(k in keys for k in self.key)
        return self.key in keys

    def __repr__(self):
        return '<Step %s>' % (self.title or self.key)


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
    """

    def __init__(self, name, title, steps, description='', aliases=(),
                 order='free', requires=None, hidden=False, see_also=()):
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

    def progress(self):
        """(done, total) for the prompt and `status`."""
        return (max(self.index, 0) + (1 if self.index >= 0 else 0),
                len(self.tutorial.steps))
