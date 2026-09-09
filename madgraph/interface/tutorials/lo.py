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
from madgraph.interface.tutorials.session import Step, Tutorial

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


def intro(interface=None):
    """The welcome text, which depends on whether madspace is already here."""

    head = """
Welcome. This tutorial takes you from nothing to a leading-order event sample,
in about six commands, and points you at the tutorial that goes deeper at each
step along the way.

As you go: `hint` and `solution` print the command a step expects -- they never
run it for you, you always type it. `repeat` prints the step again, `skip`
moves on, `tutorial status` shows how far you have got, and `tutorial stop`
leaves at any time.
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

Step('generate', """
Look at what MG5 printed back: it added `QED=0` on its own. With no coupling
orders given, it picks the combination with the most QCD vertices, because that
is nearly always the dominant one -- a sensible guess, but a guess.

That is one line of a much larger grammar. Coupling orders, interference-only
selections, required and forbidden s-channels, decay chains, polarisation and
NLO all go in the same process line; `tutorial syntax` walks through the lot.

Before generating anything, it is worth a look at the diagrams:
%(p)s display diagrams
""" % {'p': P},
     title='generate a process',
     hint="A space between every particle name, and `>` separates initial from final state.",
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

Step('output', """
You now have a directory called %(run)s.

Note what you did *not* have to say: `output` with no format produces the MG7
output, driven by madspace -- that is the default. What is inside:

  Cards/run_card.toml    beams, cuts, number of events, integrator settings
  Cards/param_card.dat   masses, widths and couplings
  SubProcesses/          the generated matrix elements
  bin/generate_events    the runner, which `launch` calls for you

Other things `output` can make, each with a tutorial of its own:
  `output madevent %(run)s`     the MG5-compatible directory layout
  `output standalone %(run)s`   the matrix element as a callable, no events

Now run it:
%(p)s launch %(run)s

A word on what happens next, because MG7 hands over to a separate program
here. `launch` starts `bin/generate_events` as its own process: it will ask you
whether to edit the cards, and for a first run the answer is to change nothing
-- just accept the defaults. If madspace was not built earlier, this is where
it gets built, so the first run takes longer than the ones after it. When the
run finishes you come back here and the tutorial picks up again.

(To stop a long run and carry on with the tutorial, press Ctrl-C.)
""" % {'p': P, 'run': RUN},
     title='produce the output',
     hint="`output NAME` with no format gives you the default MG7 output.",
     solution='launch %s' % RUN),

Step('launch', """
That is a full leading-order event sample.

What you got, and where:
  * the cross section and its uncertainty, printed at the end of the run;
  * the events themselves, an LHE file under `%(run)s/Events/`;
  * the banner at the top of that file, which records every card and every
    setting used -- it is the honest record of how the numbers were made;
  * an HTML summary in the run directory, which you can open in a browser.

Two commands worth knowing here. `open` reaches anything in the output
directory without you typing the whole path -- `open Cards/run_card.toml` shows
the settings the run actually used, `open index.html` the summary. And
`history` writes down everything you typed, so you can do this again without
having to remember it:

%(p)s history my_first_run.dat
""" % {'p': P, 'run': RUN},
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
