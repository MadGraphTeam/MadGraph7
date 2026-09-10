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
                                                  output_name)

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


def _lhapdf_note(interface=None):
    """The theory-uncertainty caveat, matched to whether LHAPDF is here.

    systematics.py imports the python lhapdf module to do the scale and PDF
    variations. Without it the run still gives a cross section and its
    integration error -- the theory uncertainty is simply absent, and the run
    log says why rather than failing.
    """

    if lhapdf_configured(interface):
        return ("Those variations need LHAPDF, which this MG7 has configured, "
                "so you should see\nthem. If the block is missing, the run log "
                "will say what went wrong.")
    return ("**You will not get the second block here.** Those variations are "
            "computed by\nsystematics.py, which needs LHAPDF, and this MG7 has "
            "no working lhapdf-config.\nThe run still succeeds and still gives "
            "you a cross section and its integration\nerror -- you just get no "
            "theory uncertainty with it. To fix that:\n\n"
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

That matters because it is the first thing to make explicit when a result
surprises you -- and it is one line of a much larger grammar. Coupling orders,
interference-only selections, required and forbidden s-channels, decay chains,
polarisation and NLO all go in the same process line; `tutorial syntax` walks
through the lot.

Before generating anything, it is worth a look at the diagrams:
%(p)s display diagrams
""" % {'p': P, 'orders': describe_applied_orders(interface)},
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

Step('output', lambda interface: """
You now have a directory called %(run)s.

Note what you did *not* have to say: `output` with no format produces the MG7
output, driven by madspace -- that is the default. What is inside:

  Cards/run_card.toml    beams, cuts, number of events, integrator settings
  Cards/param_card.dat   masses, widths and couplings
  SubProcesses/          the generated matrix elements
  bin/generate_events    the runner, which `launch` calls for you

Other things `output` can make, each with a tutorial of its own (give them a
different name so they sit beside this one):
  `output madevent DIRNAME`     the MG5-compatible directory layout
  `output standalone DIRNAME`   the matrix element as a callable, no events

Now run it:
%(p)s launch %(run)s

A word on what happens next, because MG7 hands over to a separate program
here. `launch` starts `bin/generate_events` as its own process: it will ask you
whether to edit the cards, and for a first run the answer is to change nothing
-- just accept the defaults. If madspace was not built earlier, this is where
it gets built, so the first run takes longer than the ones after it. When the
run finishes you come back here and the tutorial picks up again.

(To stop a long run and carry on with the tutorial, press Ctrl-C.)
""" % {'p': P, 'run': output_name(interface, RUN)},
     title='produce the output',
     hint="`output NAME` with no format gives you the default MG7 output.",
     solution=lambda interface: 'launch %s' % output_name(interface, RUN)),

Step('launch', lambda interface: """
That is a full leading-order event sample.

**The cross section comes with two different uncertainties**, and they are not
interchangeable.

The **statistical** one rides along with the cross section itself, in the
`Result:` row of the summary box and in the survey log lines:

    Result:   503.1(1.4)

It is the Monte-Carlo integration error and nothing more. It falls like
1/sqrt(N), so asking for more events shrinks it, and it says nothing whatever
about physics -- only about how long you ran.

The **theoretical** one comes from varying the calculation and is reported
separately, at the end, as asymmetric percentages of the central value:

    # original cross-section: 503.1
    #     scale variation: +12.4%% -9.6%%
    #     PDF variation: +2.1%% -2.1%%

Scale variation moves the renormalisation and factorisation scales (by default
x0.5, x1 and x2 each, from `[postprocessing] systematics_mur` and
`systematics_muf`); PDF variation runs the set's error members. This is the one
that goes in a paper, and no amount of extra events will shrink it -- at LO it
is usually far the larger of the two.

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
