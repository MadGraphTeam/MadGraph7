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
"""The `mlm` tutorial -- MLM merging with the Pythia8 shower.

Generates t t~ with up to one extra jet, showers it with Pythia8 and opens the
differential jet rate (DJR) plots that MG7 makes after an MLM shower.
"""

from __future__ import absolute_import

import glob
import os
import sys

import madgraph.interface.tutorials as tutorials
from madgraph.interface.tutorials.session import (Step, Tutorial,
                                                  counts_line, output_name,
                                                  pythia8_available, run_line)

P = 'MG7>'
RUN = 'MY_MLM_RUN'
DOCS = 'https://docs.madgraph.org/latest'
PYTHIA_DOCS = 'https://pythia.org/latest-manual/JetMatching.html'
DJR_PLOT = 'central_qCut_djr_plots.pdf'


def _process_dir(interface):
    """The directory the last `output` made, or None."""

    done = getattr(interface, '_done_export', None)
    return done[0] if done else None


def _shown(path):
    """`path` as the reader types it at the prompt: relative when it is below
    the current directory."""

    try:
        relative = os.path.relpath(path)
    except Exception:
        return path
    return path if relative.startswith('..') else relative


def _latest_run(interface):
    """The newest Events/<run> directory of the output, or None."""

    try:
        events = os.path.join(_process_dir(interface), 'Events')
        runs = [os.path.join(events, name) for name in os.listdir(events)]
        runs = [d for d in runs if os.path.isdir(d)]
        return max(runs, key=os.path.getmtime) if runs else None
    except Exception:
        return None


def _found_plot_dir(interface):
    """The newest <tag>_PY8_plots directory holding the DJR plot, or None."""

    try:
        found = glob.glob(os.path.join(_process_dir(interface), 'HTML', '*',
                                       '*_PY8_plots', DJR_PLOT))
    except Exception:
        return None
    if not found:
        return None
    return os.path.dirname(max(found, key=os.path.getmtime))


def _shower_ran(interface):
    """Whether the last run's shower finished: it leaves the DJR data (or the
    showered events) even when no plot could be drawn from them."""

    run = _latest_run(interface)
    if not run:
        return False
    return any(glob.glob(os.path.join(run, pattern)) for pattern in
               ('*_djrs.dat*', '*_pythia8_events.hepmc*'))


def _not_drawn(interface):
    """What to say when the shower finished but left no DJR plot: drawing
    it needs matplotlib or gnuplot, and neither worked."""

    run = _latest_run(interface)
    data = sorted(glob.glob(os.path.join(run, '*_djrs.dat*')))
    text = ("**The shower finished, but MG7 could not draw the DJR plots.**\n"
            "Drawing them needs matplotlib or gnuplot, and the run log says "
            "which one\nfailed.")
    if data:
        text += " The DJR data are in\n\n  %s\n" % _shown(data[-1])
    else:
        text += "\n"
    return text + ("\nInstall matplotlib (`pip install matplotlib`) and "
                   "launch again with the same\nanswers to get the plots. "
                   "Or read on with `skip`: the next step explains\nwhat "
                   "the plots show.\n")


def _plot_dir(interface):
    """Where the DJR plots are, as typed at the prompt.

    Before they exist, where the next run puts them: run_01 for a new output,
    one run further after a shower that failed.
    """

    found = _found_plot_dir(interface)
    if found:
        return _shown(found)
    try:
        events = os.path.join(_process_dir(interface), 'Events')
        count = len([d for d in os.listdir(events)
                     if os.path.isdir(os.path.join(events, d))])
        run = 'run_%02d' % (count + 1)
    except Exception:
        run = 'run_01'
    return os.path.join(output_name(interface, RUN), 'HTML', run,
                        'tag_1_PY8_plots')


def _open_plot(interface=None):
    opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
    return 'shell %s %s' % (opener, os.path.join(_plot_dir(interface),
                                                 DJR_PLOT))


def _launch(interface=None):
    return 'launch %s' % output_name(interface, RUN)


def _pythia8_note(interface):
    if pythia8_available(interface):
        return ''
    return ("\n**Pythia8 is not installed yet.** Install it before you go on.\n"
            "It takes a few minutes.\n\n  %s install pythia8\n" % P)


def _pythia8data_mismatch(interface):
    """The PYTHIA8DATA of the environment, when it is not the data of the
    Pythia8 that MG7 runs. main164 then stops on a version mismatch."""

    data = os.environ.get('PYTHIA8DATA')
    if not data or not pythia8_available(interface):
        return None
    try:
        root = os.path.realpath(interface.options['pythia8_path'])
        if os.path.realpath(data).startswith(root + os.sep):
            return None
    except Exception:
        return None
    return data


def _pythia8data_note(interface):
    data = _pythia8data_mismatch(interface)
    if not data:
        return ''
    return ("\n**PYTHIA8DATA is set** to a different Pythia8.\n\n  %s\n\n"
            "The shower stops if its version does not match. Leave MG7, run\n"
            "`unset PYTHIA8DATA` in your shell, and start again.\n" % data)


def _version_mismatch(interface):
    """Whether the last shower stopped on a Pythia8 data version mismatch."""

    run = _latest_run(interface)
    logs = glob.glob(os.path.join(run, 'PY8_parallelization', '*',
                                  'PY8_log.txt')) if run else []
    for log in logs:
        try:
            with open(log) as handle:
                if 'unmatched version numbers' in handle.read():
                    return True
        except Exception:
            continue
    return False


def _shower_failed(interface):
    """What to say when the run left no DJR plot."""

    run = _latest_run(interface)
    crash = os.path.join(run, 'Pythia8_shower_crash.log') if run else None
    text = "**The shower did not finish**, so there are no DJR plots.\n"
    if crash and os.path.exists(crash):
        text += "The error is in\n\n  %s\n" % _shown(crash)
    if _version_mismatch(interface):
        return text + """
Pythia8 found data files of another version. PYTHIA8DATA points to them.
Leave MG7, run `unset PYTHIA8DATA` in your shell, and start MG7 again.
Then `tutorial mlm` and `skip 4` bring you back to the launch.
"""
    return text + ("\nThe Pythia8 logs are in the run's PY8_parallelization "
                   "directory.\nFix the problem and launch again with the "
                   "same answers. Once the plot exists, open it.\n")


def _plots_found(interface):
    if not _found_plot_dir(interface):
        if _shower_ran(interface):
            return _not_drawn(interface)
        return _shower_failed(interface)
    return """The Pythia8 summary at the end of the run shows two more numbers.

  Matched cross-section                  the cross section of the merged
                                         sample
  Nb of events after matching/merging    the events the matching kept

The matching drops the events whose jets do not match the partons. The
showered events are in `tag_1_pythia8_events.hepmc.gz`, next to the parton
level events.

After an MLM shower, MG7 also plots the **differential jet rates** (DJRs).
They are the standard check of a merged sample. The plots are in

  %s

`index.html` there lists every plot. Each one also comes as an HwU file and
as a Python script, if you want to restyle it. Open the main plot.
""" % _plot_dir(interface)


def _parton_result(interface):
    found = run_line(interface)
    return ('The parton-level cross section is %s.\nIt adds up both jet '
            'samples.\n\n' % found) if found else ''


LAUNCH_HINT = """
Type these lines, then press Enter.

  shower=Pythia8      shower the events with Pythia8
  set events 10000    fewer events than the default, for a quick run

The [merging] section of run_card.toml holds the MLM settings.
"""


def _launch_progress(interface, line):
    words = (line or '').replace('=', ' ').lower().split()
    if words[:2] == ['shower', 'pythia8']:
        return ("The shower is on. Set the number of events too, then "
                "press Enter.")
    if words[:2] == ['set', 'events']:
        return "Now press Enter to start the run."
    return None


tutorial = Tutorial(
    name='mlm',
    title='MLM merging and DJR plots',
    description='merge t t~ + 0 and 1 jet with MLM, shower with Pythia8 and '
                'check the DJR plots',
    aliases=('merging',),
    section='advanced',
    ai_generated=False,
    order='sequence',
    see_also=('lo', 'gridpack', 'decays'),
    steps=[

Step('tutorial', lambda interface: """
This tutorial assumes you have run a process before. If not, start with
`tutorial lo`. It covers **MLM merging**.

The matrix element describes hard, well separated jets well. The parton
shower describes soft and collinear emissions well. A merged sample takes
each jet from where it is described best. It combines processes with
different numbers of jets, and the shower adds the rest.

Both can produce the same jet, so a merged sample has to remove the double
counting. MLM does this in the shower. An event is kept only if its jets
match the partons of the matrix element.

We merge t t~ with up to one extra jet. Each jet multiplicity is a process
of its own. Start with the one without a jet.
%(pythia8)s%(pythia8data)s
%(p)s generate p p > t t~
""" % {'p': P, 'pythia8': _pythia8_note(interface),
       'pythia8data': _pythia8data_note(interface)},
     title='welcome',
     solution='generate p p > t t~'),

Step('generate', lambda interface: """
%(counts)sNow add the same process with one more jet.

%(p)s add process p p > t t~ j
""" % {'p': P, 'counts': counts_line(interface)},
     title='the sample without a jet',
     solution='add process p p > t t~ j'),

Step('add_process', lambda interface: """
%(counts)sThe two lines you typed are the two **jet samples**. Sample 0 is
t t~, sample 1 is t t~ with one jet. Each sample has several subprocesses,
which is why the count is higher.

The second process only adds a jet to the first. MG7 recognises this and
sets up an MLM sample on its own. Write the output.

%(p)s output %(run)s
""" % {'p': P, 'run': RUN, 'counts': counts_line(interface)},
     title='the sample with one jet',
     solution='output %s' % RUN),

Step('output', lambda interface: """
The run card already holds the MLM settings.

  dynamical_scale_choice = "mlm"   the scales follow the clustering of each
                                   event, as MLM needs
  xqcut = 30.0                     the smallest kt of the matrix-element
                                   jets, in GeV
  auto_ptj_mjj = true              ties the jet cuts to xqcut

`xqcut` is in the `[merging]` section. It sets where the matrix element takes
over from the shower. Jets above it come from the matrix element.

The matching itself happens in Pythia8. It uses its own scale, `qCut`. MG7
sets it to 1.5 times `xqcut` unless you choose another value in the Pythia8
card.

Launch, turn on the shower at the card question, and lower the number of
events. The hint there lists the lines to type.

%(p)s %(launch)s
""" % {'p': P, 'launch': _launch(interface)},
     title='the MLM settings',
     question_hint=LAUNCH_HINT,
     question_progress=_launch_progress,
     hint="At the card question, `shower=Pythia8` and `set events 10000`, "
          "then Enter.",
     solution=_launch),

Step('launch', lambda interface: """
%(result)s%(found)s%(open)s""" % {
       'result': _parton_result(interface), 'found': _plots_found(interface),
       # after `skip 4` this step shows again, with the right path
       'open': '' if not _found_plot_dir(interface) and
               (_version_mismatch(interface) or _shower_ran(interface))
               else '\n%s %s\n' % (P, _open_plot(interface))},
     title='the shower and the merging',
     hint="`shell open PATH` opens a file, or `shell xdg-open PATH` on Linux.",
     solution=_open_plot),

Step('shell', lambda interface: """
`d01` is the kt scale where a showered event goes from zero to one jet,
`d12` from one to two, and so on. The plots show their log10. Jet sample 0
dominates at low values, where the shower makes the jet. Jet sample 1
dominates at high values, where the matrix element makes it. Their **sum**
should be smooth, in particular around log10(qCut), about 1.65 here. A step
or a dip means the merging is not under control. Then change the merging
scales and run again.

  set xqcut 40                  at the card question. `qCut` follows it.
  set JetMatching:qCut 60       sets `qCut` in the Pythia8 card directly.
                                Keep it at least 1.5 times `xqcut`.

The highest jet sample is inclusive. Here, jet sample 1 also contains events
with a second jet, from the shower. For more jets, add the next process, for
example `add process p p > t t~ j j`.

In short:

  * Add one process per jet multiplicity. MG7 then sets up MLM on its own.
  * `xqcut` in the run card and `qCut` in the Pythia8 card set the merging
    scales.
  * `shower=Pythia8` showers and matches the events. The DJR plots are in
    `HTML/<run>/<tag>_PY8_plots`.

The documentation has every option.

  the [merging] section of the run card
  %(docs)s/run_card.html#merging

  the Pythia8 jet matching settings
  %(pythia)s

%(see_also)s

Leave with `tutorial stop`.
""" % {'docs': DOCS, 'pythia': PYTHIA_DOCS,
       'see_also': tutorials.where_next()},
     title='reading the DJR plots'),

    ],
)
