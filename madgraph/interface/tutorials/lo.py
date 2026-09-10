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
"""The `lo` tutorial -- from a cold start to events on disk.

The shortest honest path to a leading-order event sample, on madspace (which is
what `output` produces by default), with a signpost at each step to the
tutorial that goes deeper.  Everything the pre-2026 version taught after
`launch` -- loading a model, restricting it, defining multiparticles -- has
moved to the tutorials that own it.
"""

from __future__ import absolute_import

import os

import madgraph
import madgraph.interface.tutorials as tutorials
from madgraph.interface.tutorials.session import (Step, Tutorial,
                                                  describe_applied_orders,
                                                  lhapdf_configured,
                                                  output_name,
                                                  total_diagrams)

P = 'MG7>'
RUN = 'MY_FIRST_LO_RUN'


def madspace_is_installed():
    """True if this MG7 already has a usable madspace build.

    Mirrors what the generated bin/generate_events checks before bootstrapping
    one for itself (see iolibs/template_files/mg7/launch.py).
    """

    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(
            madgraph.__file__)))
    except Exception:
        return False
    return os.path.isdir(os.path.join(root, 'madspace', 'install', 'madspace'))


def _detour_text(interface=None):
    """The optional QED<=2 side-trip: say what changed, then send them back.

    Reached only if the user takes it; typing `display diagrams` at the
    previous step jumps straight past this one.
    """

    before = getattr(interface, '_tutorial_lo_ndiag', None)
    now = total_diagrams(interface)

    if before and now and now != before:
        count = ("You had **%d** diagrams a moment ago and now you have "
                 "**%d**." % (before, now))
    elif now:
        count = "You now have **%d** diagrams." % now
    else:
        count = ""

    return """
%(count)s

The new ones are all in the quark-antiquark subprocess. Without QED vertices
it can only go through a gluon; allowing two lets it go through a photon or a
Z as well, so `q q~ > t t~` picks up the two s-channel electroweak diagrams.
The gluon-gluon subprocess is unchanged -- there is no electroweak way to make
a top pair out of two gluons at this order.

Whether that matters is a physics question, not a syntax one. Here the
electroweak contribution is tiny next to QCD, which is exactly why MG5's search
left it out. For a process where it is not tiny -- anything with a Z or a W in
the final state -- leaving it to the search is how you quietly compute the
wrong thing.

That is the detour. Back to the main line:
%(p)s display diagrams
""" % {'count': count, 'p': P}


def _lhapdf_note(interface=None):
    """The theory-uncertainty caveat, matched to whether LHAPDF is here.

    madspace evaluates the PDF members itself, but it still has to *have* the
    set: it resolves and, if need be, downloads it through lhapdf-config
    (mg7/launch.py, ensure_pdf_set). Without one the scale variations still
    work -- they need no new PDF -- but the error members cannot be found, and
    the run says so rather than failing.
    """

    if lhapdf_configured(interface):
        return ("The PDF half needs the error set on disk, which madspace "
                "resolves through\nlhapdf-config -- configured here, and it "
                "will download the set if it has to.\nIf a row is missing, the "
                "run log says why.")
    return ("**Expect the PDF row to be missing here.** madspace evaluates the "
            "PDF members\nitself, but it still has to find the set, and it "
            "does that through lhapdf-config,\nwhich this MG7 does not have. "
            "The scale variations need no new PDF and should\nstill appear; "
            "the cross section and its integration error are unaffected.\n\n"
            "  MG7> install lhapdf6\n\n"
            "and then `set lhapdf /path/to/lhapdf-config` if MG7 does not find "
            "it by itself.")


def intro(interface=None):
    """The welcome text, which depends on whether madspace is already here."""

    head = """
Welcome. This tutorial takes you from nothing to a leading-order event sample,
in about six commands, and points you at the tutorial that goes deeper at each
step along the way.

As you go: `hint` and `solution` print the command a step expects -- they never
run it for you, you always type it. `repeat` prints the step again, `skip`
moves on, `tutorial status` shows how far you have got, and `tutorial stop`
leaves at any time. `tutorial help` lists the lot.
"""

    if madspace_is_installed():
        return head + """
The integration engine, madspace, is already installed here, so we can go
straight to physics. Pick a process:
%(p)s generate p p > t t~

`p` is a multiparticle label -- shorthand for a set of particles. A space
between particle names is mandatory.
""" % {'p': P}

    return head + """
One thing first. Events are integrated by madspace, and this MG7 does not have
it built yet:
%(p)s install madspace

(You can skip this -- the run itself will build madspace the first time it
needs it. Doing it now just means you see the build separately from the
physics, which is easier to read if anything goes wrong.)
""" % {'p': P}


tutorial = Tutorial(
    name='lo',
    title='first events at leading order',
    description='from a cold start to an event sample, with madspace',
    aliases=('MadGraph5',),
    section='basic',
    ai_generated=False,
    order='sequence',
    see_also=('syntax', 'madevent', 'mg7', 'model', 'bsm', 'standalone',
              'nlo', 'run', 'decays', 'checks', 'exercises'),
    steps=[

Step('tutorial', intro,
     title='welcome',
     hint="If madspace is already installed, go straight to `generate`.",
     solution='generate p p > t t~'),

Step('install', """
madspace is in place. Now pick a process:
%(p)s generate p p > t t~

`p` is a multiparticle label -- shorthand for a set of particles. A space
between particle names is mandatory.
""" % {'p': P},
     title='install madspace',
     solution='generate p p > t t~'),

Step('generate', lambda interface: """
Look at what MG5 printed back. You gave it no coupling orders, so it chose
some for you.

%(orders)s

You can ask for the electroweak diagrams back by allowing QED vertices:

%(p)s generate p p > t t~ QED<=2

Note that `tutorial syntax` walks through the various ways to generate
diagrams.

The typical next step is to inspect the generated diagrams to check them:

%(p)s display diagrams
""" % {'p': P, 'orders': describe_applied_orders(interface)},
     title='generate a process',
     hint="A space between every particle name, and `>` separates initial from final state.",
     setup=lambda interface: setattr(interface, '_tutorial_lo_ndiag',
                                     total_diagrams(interface)),
     solution='display diagrams'),

Step('generate', lambda interface: _detour_text(interface),
     title='the electroweak diagrams (detour)',
     hint="Nothing to do here -- `display diagrams` picks the main line back up.",
     solution='display diagrams'),

Step('display', """
That opens the diagrams in a viewer (or writes them, depending on your setup).
Two minutes here saves an afternoon: a process with far more or far fewer
diagrams than you expected usually means the process line said something other
than what you meant.

`tutorial checks` covers the heavier validation -- `check gauge`,
`check lorentz`, `check permutation` -- for when the diagrams look right but
the number does not.

Now produce the output:
%(p)s output %(run)s
""" % {'p': P, 'run': RUN},
     title='look at the diagrams',
     solution='output %s' % RUN),

Step('output', lambda interface: """
You now have a directory called %(run)s.

Note that `output` can take a format: `output FORMAT PATH`. No format is our
default output for event generation.

Other things `output` can make, each with a tutorial of its own (give them a
different name so they sit beside this one):
  `output madevent DIRNAME`     the MG5-compatible directory layout
  `output standalone DIRNAME`   the matrix element as a callable, no events

Now run it:
%(p)s launch
or
%(p)s launch %(run)s

-- with no argument it takes the output you just made.

Then it runs. If madspace was not built earlier this is where it gets built, so
the first run takes longer than the ones after it.

(To stop a long run and carry on with the tutorial, press Ctrl-C.)
""" % {'p': P, 'run': output_name(interface, RUN)},
     title='produce the output',
     hint="`output NAME` with no format gives you the default MG7 output.",
     solution=lambda interface: 'launch %s' % output_name(interface, RUN)),

Step('launch', lambda interface: """
That is a full leading-order event sample.

**The question you just answered** had two blocks. The top one was the programs
to run after generation -- parton shower, detector simulation, analysis,
MadSpin for decays with spin correlations, reweighting -- where a number or a
name toggles one, and `Not Avail.` means it is not installed here. The bottom
one was the cards: always `param_card.dat` (masses, widths, couplings) and
`run_card.toml` (beams, cuts, number of events, integrator settings), plus one
for each program switched on. A number opens a card in your editor,
`set KEY VALUE` changes a single parameter without one, and a path to an
existing card or banner reuses it wholesale.

**The cross section comes with two different uncertainties**, and they are not
interchangeable.

The **statistical** one rides along with the cross section itself, in the
`Result:` row of the summary box and in the survey log lines:

    Result:   503.1(1.4)

It is the Monte-Carlo integration error and nothing more. It falls like
1/sqrt(N), so asking for more events shrinks it, and it says nothing whatever
about physics -- only about how long you ran.

The **theoretical** one comes from varying the calculation, and gets a box of
its own at the end of the run:

    Systematics
    Variations per event:    109 (9 scale, 101 PDF members)
    PDF set:                 NNPDF23_lo_as_0130_qed, replicas
    Original cross-section:  503.1 pb
    Scale variation:         +12.4%%    -9.6%%
    PDF variation:           +2.1%%    -2.1%%

Scale variation moves the renormalisation and factorisation scales (x0.5, x1
and x2 each by default, from `[systematics] mur` and `muf`); PDF variation runs
the set's error members. This is the one that goes in a paper, and no amount of
extra events will shrink it -- at LO it is usually far the larger of the two.

madspace computes these while it writes the events, so they cost one pass and
no extra integration. Every event carries its variation weights, and the
per-variation cross sections land in `info.json` next to the events.

%(lhapdf)s

The rest of what you got:
  * the events themselves, an LHE file under `%(run)s/Events/`;
  * the banner at the top of that file, which records every card and every
    setting used -- it is the honest record of how the numbers were made. The
    per-event variation weights live there too;
  * an HTML summary in the run directory, which you can open in a browser.

Two commands worth knowing here. `open` reaches anything in the output
directory without you typing the whole path -- `open Cards/run_card.toml` shows
the settings the run actually used, `open index.html` the summary. And
`history` writes down everything you typed, so you can do this again without
having to remember it:

%(p)s history my_first_run.dat
""" % {'p': P, 'run': output_name(interface, RUN),
       'lhapdf': _lhapdf_note(interface)},
     title='run it',
     hint="`history FILE` saves the session; `open FILE` shows a file from the output.",
     solution='history my_first_run.dat'),

Step(('history', 'open', 'display'), lambda interface: """
That is the whole path: generate, look, output, run, read.

You now have a file that replays the whole session:

  MG7> import command my_first_run.dat

or, from the shell, `./bin/madgraph my_first_run.dat`. That is how a run stops
being something you remember and starts being something you can repeat.

Two more things worth knowing:
  * any shell command works from here -- `shell ls`, or just `!ls`;
  * `help COMMAND` prints the syntax for any command, and `help` on its own
    lists them all.

Where to go next:
%(see_also)s

Leave tutorial mode with `tutorial stop`. Thanks for using the tutorial!
""" % {'see_also': tutorials.see_also_block(
           ['syntax', 'model', 'bsm', 'run', 'decays', 'madevent', 'mg7',
            'standalone', 'nlo', 'madloop', 'checks', 'exercises'])},
     title='where to go next'),

    ],
)
