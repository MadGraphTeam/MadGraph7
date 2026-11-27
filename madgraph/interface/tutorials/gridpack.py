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
"""The `gridpack` tutorial -- gridpacks and cluster runs.

Creates a gridpack, runs it, and then parallelises a run with cluster mode.
The cluster part uses `type = "local_parallel"`, which runs the same batch jobs
as processes on this machine, so the tutorial works without a scheduler.
"""

from __future__ import absolute_import

import os

import madgraph.interface.tutorials as tutorials
from madgraph.interface.tutorials.session import (Step, Tutorial,
                                                  counts_line, output_name,
                                                  run_line)

P = 'MG7>'
RUN = 'MY_GRIDPACK_RUN'
DOCS = 'https://docs.madgraph.org/latest'


def _process_dir(interface):
    """The directory the last `output` made, or None."""

    done = getattr(interface, '_done_export', None)
    return done[0] if done else None


def _latest_run(interface):
    """The newest Events/<run> directory of the output, or None."""

    try:
        events = os.path.join(_process_dir(interface), 'Events')
        runs = [os.path.join(events, name) for name in os.listdir(events)]
        runs = [d for d in runs if os.path.isdir(d)]
        return max(runs, key=os.path.getmtime) if runs else None
    except Exception:
        return None


def _shown(path):
    """`path` as the reader types it at the prompt: relative when it is below
    the current directory."""

    try:
        relative = os.path.relpath(path)
    except Exception:
        return path
    return path if relative.startswith('..') else relative


def _gridpack(interface):
    """The gridpack of the last run, as typed at the prompt.

    Falls back to where the first run puts it, so the step still reads
    sensibly before anything has run.
    """

    run = _latest_run(interface)
    if run is not None:
        for name in ('gridpack', 'gridpack.tar.gz'):
            if os.path.exists(os.path.join(run, name)):
                return _shown(os.path.join(run, name))
    return os.path.join(output_name(interface, RUN), 'Events', 'run_01',
                        'gridpack')


def _run_gridpack(interface=None):
    return ('shell %s/bin/generate_events --events 1000 --seed 42'
            % _gridpack(interface))


def _launch(interface=None):
    return 'launch %s' % output_name(interface, RUN)


def _gridpack_found(interface):
    """What the run left behind, in the reader's own path."""

    gridpack = _gridpack(interface)
    if gridpack.endswith('.tar.gz'):
        return ("The gridpack is packed, because `compress` is on.\n\n  %s\n\n"
                "Unpack it with `shell tar xzf %s -C %s` before you run it.\n"
                % (gridpack, gridpack, os.path.dirname(gridpack)))
    return "The gridpack is in\n\n  %s\n" % gridpack


def _cluster_result(interface):
    found = run_line(interface)
    return 'The combined cross section is %s.\n\n' % found if found else ''


FIRST_LAUNCH_HINT = """
Type this, then press Enter.

  set save_gridpack true

`update gridpack` shows the whole [gridpack] section in the card first.
"""


def _first_launch_progress(interface, line):
    words = (line or '').replace('=', ' ').split()
    if words[:2] in (['set', 'save_gridpack'],
                     ['set', 'gridpack.save_gridpack']):
        return "Now press Enter to start the run."
    if words[:2] == ['update', 'gridpack']:
        return ("The [gridpack] section is in the card now. "
                "Type `set save_gridpack true`, then press Enter.")
    return None


CLUSTER_LAUNCH_HINT = """
Type these lines, then press Enter.

  set cluster.enable true            turn on cluster mode
  set cluster.type local_parallel    run the jobs on this machine
  set events 10000                   events in total
  set cluster.events_per_job 2500    four generation jobs

`set cluster.parallel_jobs 2` limits the jobs running at once. The
default, -1, uses every core.
"""


def _cluster_launch_progress(interface, line):
    words = (line or '').replace('=', ' ').split()
    if words[:2] == ['set', 'cluster.events_per_job']:
        return "That is everything. Press Enter to start the run."
    return None


GRIDPACK_ONLY_HINT = """
Cluster mode is still on. Type these lines, then press Enter.

  set cluster.mode gridpack_only    stop after the training
  set madnis.enable true            train MadNIS
  set madnis.train_batches 2000     a short training

The [gridpack] options also apply to the merged gridpack.
"""


def _gridpack_only_progress(interface, line):
    words = (line or '').replace('=', ' ').split()
    if words[:2] == ['set', 'madnis.train_batches']:
        return "That is everything. Press Enter to start the run."
    return None


tutorial = Tutorial(
    name='gridpack',
    title='gridpacks and cluster runs',
    description='create and run a gridpack, and parallelise a run with '
                'cluster mode',
    aliases=('cluster',),
    section='advanced',
    ai_generated=False,
    order='sequence',
    see_also=('lo', 'mg7'),
    steps=[

Step('tutorial', """
This tutorial assumes you have run a process before. If not, start with
`tutorial lo`. It covers two ways to scale up event generation.

  * A **gridpack** is a snapshot of a finished survey and training. Only the
    event generation is left. It starts quickly, and you can run it as often
    as you like, on any machine.
  * **Cluster mode** splits a run into batch jobs that run in parallel. It is
    built on gridpacks.

We use a small process, so that every run is quick.

%(p)s generate p p > t t~
""" % {'p': P},
     title='welcome',
     solution='generate p p > t t~'),

Step('generate', lambda interface: """
%(counts)sRemember the two subprocesses, `g g > t t~` and `q q~ > t t~`.
They matter later, for cluster mode.

%(p)s output %(run)s
""" % {'p': P, 'run': RUN, 'counts': counts_line(interface)},
     title='a process with two subprocesses',
     solution='output %s' % RUN),

Step('output', lambda interface: """
The gridpack settings are in the `[gridpack]` section of
`Cards/run_card.toml`. The section is hidden by default. It appears once you
change one of its settings, and `set` works on it all the same.

With `save_gridpack` on, the run does the survey and optimises the grids as
usual. Then it **stops without generating events** and stores the gridpack.

Launch, type `set save_gridpack true` at the card question, and press Enter.

%(p)s %(launch)s
""" % {'p': P, 'launch': _launch(interface)},
     title='turn the gridpack on',
     question_hint=FIRST_LAUNCH_HINT,
     question_progress=_first_launch_progress,
     hint="`set save_gridpack true` at the card question, then Enter.",
     solution=_launch),

Step('launch', lambda interface: """
%(found)s
It is self-contained.

  bin/generate_events    the event generator
  Cards/                 grid_run_card.toml and param_card.dat
  data/                  the optimised channels
  lib/                   the compiled matrix elements
  madspace/              a compiled copy of madspace
  Events/                the output of each gridpack run

The cuts, beams, masses and the integration are fixed in the gridpack. Each
run reads `grid_run_card.toml` for what can still change, like the number of
events, the seed, the devices and the output format. Each of these also has a
command-line option.

Now run it. `shell` runs a command from this prompt. In a real study, a job
script would call the gridpack on a cluster node.

%(p)s %(run_gp)s
""" % {'p': P, 'found': _gridpack_found(interface),
       'run_gp': _run_gridpack(interface)},
     title='what is in a gridpack',
     hint="`shell PATH/bin/generate_events --events N --seed S` runs the "
          "gridpack.",
     solution=_run_gridpack),

Step('shell', lambda interface: """
That run had no survey and no training, so it was much faster than a full
run. The events are in

  %(gridpack)s/Events/run_01/events.lhe.gz

A gridpack writes LHE by default, since its events usually go to a shower.
`--output_format` selects one of the npy formats instead.

Three options matter when several jobs share a gridpack.

  --seed              must differ between jobs. The default, -1, draws a new
                      seed and records it.
  --output_dir        writes the events there instead of into the gridpack.
  --temp_output_dir   holds the large temporary files. `auto` makes a new
                      directory per run in $TMPDIR, so jobs do not collide.

`--help` lists all options.

**What goes into the gridpack** is set in the `[gridpack]` section. By
default it holds compiled code only. That keeps it small and ready to run,
but it only runs on the platform it was built on.

  include_source           ship the matrix element source instead of lib/.
                           Build it once on the target machine.
  include_madspace         ship the compiled madspace. On by default.
  include_madspace_source  ship the madspace source. It is built on the
                           first call.
  compress                 pack the gridpack into gridpack.tar.gz.

With `include_source` and `include_madspace_source`, the gridpack runs on any
machine with a compiler. Keep the default if all machines are the same.

Alternatively, install madspace system-wide on the target machines, for
example with `pip install madspace`. Then set `include_madspace = false`, and
the gridpack uses the installed copy. It must be the same madspace version
that created the gridpack. Otherwise the gridpack stops with an error.

`run_mode` sets how the run that creates the gridpack ends.

  minimal         The default. No events. Each gridpack run estimates the
                  maximum weights itself.
  fix_max_weight  No events. Each channel runs until its maximum weight is
                  known, and the gridpack stores it. All gridpack runs then
                  unweight the same way. This is slower to create, but better
                  when each job makes few events. It also adds
                  bin/event_stream.py to stream events into Python.
  regular         A normal run with events, then the gridpack.

Next comes **cluster mode**. A normal run trains the subprocesses one after
the other, in one process. Cluster mode gives each subprocess its own
training job, and all of them run in parallel. Here that means two jobs, one
for `g g > t t~` and one for `q q~ > t t~`. Further jobs then generate the
events.

A real cluster needs a scheduler. This tutorial uses `local_parallel`
instead. It runs the same jobs as separate processes on this machine.
Moving to a real cluster later is one line in the card.

Launch the same output again and turn on cluster mode at the card question.
The hint there lists the lines to type.

%(p)s %(launch)s
""" % {'p': P, 'gridpack': _gridpack(interface),
       'launch': _launch(interface)},
     title='running a gridpack, and what to put in it',
     question_hint=CLUSTER_LAUNCH_HINT,
     question_progress=_cluster_launch_progress,
     hint="At the card question, `set cluster.enable true` and "
          "`set cluster.type local_parallel`, then Enter.",
     solution=_launch),

Step('launch', lambda interface: """
%(result)sThat was a whole run, split into jobs.

  1. **Training.** One job per subprocess, here two in parallel. Each job
     runs its own survey and training and saves a gridpack.
  2. **Merging.** The two gridpacks become one with all channels.
  3. **Generation.** Several jobs run the merged gridpack, each with its own
     seed. `events_per_job` sets how many.
  4. **Combining.** The events are joined into an ordinary run in
     `%(run)s`, in the `output_format` of the run card. The cross section is
     averaged over the jobs.

The logs of all jobs are in `%(run)s/cluster/`. If a run is interrupted or a
job fails, launch again without changing the cards. The run continues and
only repeats the jobs that did not finish.

On a real cluster, set `type` to your scheduler, like slurm, condor, pbs, lsf
or sge. The same jobs are then submitted there. `[cluster_train]` and
`[cluster_deploy]` set how each pass runs. This example trains on GPUs and
generates on CPUs.

  [cluster]
  type = "slurm"
  train_extra_args = "--gres=gpu:1"
  [cluster_train]
  run.device = ["cuda"]
  [cluster_deploy]
  run.device = ["cpu"]

Put the setup of your site in `input/default_run_card_mg7.toml` once. Every
process then picks it up. The jobs need the process directory and MG7 on a
shared file system.

Often only the training should run on the cluster. The gridpack is then run
elsewhere. This is the `gridpack_only` mode.

For a large sample, a longer training also pays off. Its cost is fixed,
while every event after it is cheaper. So we also turn on MadNIS, with a
short training that finishes quickly here. For a real sample, use more
`train_batches`.

%(p)s %(launch)s
""" % {'p': P, 'launch': _launch(interface),
       'result': _cluster_result(interface),
       'run': _shown(_latest_run(interface) or
                     os.path.join(output_name(interface, RUN), 'Events',
                                  'run_02'))},
     title='a cluster run, locally',
     question_hint=GRIDPACK_ONLY_HINT,
     question_progress=_gridpack_only_progress,
     hint="At the card question, `set cluster.mode gridpack_only`, "
          "`set madnis.enable true` and `set madnis.train_batches 2000`, "
          "then Enter.",
     solution=_launch),

Step('launch', lambda interface: """
That run stopped after the merging. The MadNIS training ran in parallel, one
job per subprocess, and the result is one merged gridpack.

%(found)s
It is an ordinary gridpack. Everything about gridpacks above applies to it.
This is the usual pattern for a large sample. Train once in parallel, then
run the gridpack in as many jobs as you need.

Run it the same way as the first one.

%(p)s %(run_gp)s
""" % {'p': P, 'found': _gridpack_found(interface),
       'run_gp': _run_gridpack(interface)},
     title='training only, a merged gridpack',
     hint="`shell PATH/bin/generate_events --events N --seed S` runs the "
          "gridpack.",
     solution=_run_gridpack),

Step('shell', lambda interface: """
These events come from the merged gridpack, with the trained MadNIS
channels. They are in

  %(gridpack)s/Events/run_01/events.lhe.gz

Compare `unw. eff.` in the output with the first gridpack run. The MadNIS
channels usually unweight better, which is what makes a long training pay
off for a large sample. On a cluster, each job would run this command with
its own `--seed`.

Cluster mode stays on in the card of this output. Turn it off with
`set cluster.enable false` at the next launch. Cluster mode only generates
events. Run the shower, MadSpin, Delphes or reweighting on its events
afterwards.

In short:

  * `save_gridpack = true` turns a run into a gridpack. `[gridpack]` sets
    what goes into it and how the run ends.
  * `bin/generate_events --events N --seed S` runs a gridpack.
  * `[cluster] enable = true` splits a run into jobs. `type` picks the
    scheduler, or `local_parallel` for this machine. `mode` picks events or
    a merged gridpack.

The documentation lists every option.

  %(docs)s/gridpacks.html       gridpacks
  %(docs)s/cluster.html         cluster mode, bin/merge_gridpacks
  %(docs)s/event_stream.html    streaming events into Python

%(see_also)s

Leave with `tutorial stop`.
""" % {'gridpack': _gridpack(interface), 'docs': DOCS,
       'see_also': tutorials.where_next()},
     title='running the merged gridpack'),

    ],
)
