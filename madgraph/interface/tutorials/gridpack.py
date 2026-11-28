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

Creates a gridpack, runs it, makes one that decays its events with MadSpin,
and then parallelises a run with cluster mode.
The cluster part uses `type = "local_parallel"`, which runs the same batch jobs
as processes on this machine, so the tutorial works without a scheduler.
"""

from __future__ import absolute_import

import os
import textwrap

import madgraph.interface.extended_cmd as extended_cmd
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


def _run_order(path):
    """Sort key of a run directory, newest last: its number (run_NN), then
    when it was created. Not its modification time: the info.json of a run
    can be written again once a later run has ended."""

    stat = os.stat(path)
    suffix = os.path.basename(path).rsplit('_', 1)[-1]
    return (int(suffix) if suffix.isdigit() else -1,
            getattr(stat, 'st_birthtime', stat.st_mtime))


def _latest_run(interface):
    """The newest Events/<run> directory of the output, or None."""

    try:
        events = os.path.join(_process_dir(interface), 'Events')
        runs = [os.path.join(events, name) for name in os.listdir(events)]
        runs = [d for d in runs if os.path.isdir(d)]
        return max(runs, key=_run_order) if runs else None
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


def _gridpack_dir(interface=None):
    """The gridpack directory to run, as typed at the prompt: the unpacked
    directory when `compress` packed the gridpack (the archive holds the
    `gridpack` directory, so `tar xzf ... -C run_XX` unpacks it there)."""

    gridpack = _gridpack(interface)
    if gridpack.endswith('.tar.gz'):
        return gridpack[:-len('.tar.gz')]
    return gridpack


def _run_gridpack(interface=None):
    return ('shell %s/bin/generate_events --events 1000 --seed 42'
            % _gridpack_dir(interface))


def _events_in(directory):
    """The events file a gridpack run left in `directory`, or None: the
    decayed events when the gridpack carries MadSpin."""

    for name in ('events_decayed.lhe.gz', 'events.lhe.gz', 'events.lhe',
                 'events.npy'):
        if os.path.exists(os.path.join(directory, name)):
            return os.path.join(directory, name)
    return None


def _gridpack_events(interface=None):
    """The events of the newest run of the gridpack, as typed at the prompt;
    where the first run puts them before anything has run."""

    events = os.path.join(_gridpack_dir(interface), 'Events')
    try:
        runs = [os.path.join(events, name) for name in os.listdir(events)]
        runs = [d for d in runs if os.path.isdir(d)]
        for run in sorted(runs, key=_run_order, reverse=True):
            found = _events_in(run)
            if found:
                return _shown(found)
    except Exception:
        pass
    return os.path.join(events, 'run_01', 'events.lhe.gz')


def _gridpack_events_found(interface=None):
    """The events of the newest run of the gridpack, or None when no run of
    it wrote any."""

    found = _gridpack_events(interface)
    return found if os.path.exists(found) else None


def _gridpack_made(interface, line=None):
    """Gate: the run made the gridpack the lessons after it are about."""

    run = _latest_run(interface)
    if run is not None and any(
            os.path.exists(os.path.join(run, name))
            for name in ('gridpack', 'gridpack.tar.gz')):
        return None
    return ("""
That run made no gridpack%s, so the tutorial stays here. The output of the
run above says why. Launch again, and check that `save_gridpack` is true at
the card question:

%s %s""" % (' in %s' % _shown(run) if run else '', P, _launch(interface)))


def _gridpack_ran(interface, line=None):
    """Gate: the gridpack run wrote its events."""

    if _gridpack_events_found(interface):
        return None
    return ("""
That gridpack run wrote no events in %s, so the tutorial stays here. Its
output above says why. A path typed wrong is the usual reason: the gridpack
is %s. Run it again:

%s %s""" % (_shown(os.path.join(_gridpack_dir(interface), 'Events')),
            _gridpack_dir(interface), P, _run_gridpack(interface)))


def _grid_run_card(interface=None):
    """The gridpack's own run card, as typed at the prompt (in the unpacked
    directory when `compress` packed the gridpack)."""

    return os.path.join(_gridpack_dir(interface), 'Cards',
                        'grid_run_card.toml')


def _cores_here():
    return os.cpu_count() or 2


def _cores_word(count):
    return '%d core%s' % (count, '' if count == 1 else 's')


def _half_cores():
    return max(1, _cores_here() // 2)


JOBS = ('JOB_1', 'JOB_2')


def _run_parallel(interface=None):
    """Two gridpack jobs at once, half of the cores each, then wait."""

    command = ('%s/bin/generate_events --events 10000 --seed %d '
               '--cpu_thread_pool_size %d --output_dir %s > %s.log')
    gridpack = _gridpack_dir(interface)
    # no `;`: the MG7 prompt splits a line on it, and `wait` would be read
    # as an MG7 command
    return 'shell %s & %s & wait' % tuple(
        command % (gridpack, 42 + i, _half_cores(), job, job)
        for i, job in enumerate(JOBS))


def _job_seed(number):
    """The seed job `number` of the parallel run used, from its info.json,
    or None."""

    import json
    try:
        with open(os.path.join(JOBS[number - 1], 'info.json')) as f:
            return json.load(f)['seed']
    except Exception:
        return None


def _jobs_ran(interface, line=None):
    """Gate: both parallel jobs wrote events, with different seeds."""

    missing = [job for job in JOBS if not _events_in(job)]
    if missing:
        return ("""
%s wrote no events, so the tutorial stays here. Its log, %s.log, says why.
Run the two jobs again:

%s %s""" % (' and '.join(missing), missing[0], P, _run_parallel(interface)))
    seeds = [_job_seed(1), _job_seed(2)]
    if seeds[0] is not None and seeds[0] == seeds[1]:
        return ("""
Both jobs ran with seed %s, so JOB_2 holds exactly the events of JOB_1: the
same seed gives the same events. Two jobs make one larger sample only with
different seeds. Run them again, with `--seed 42` and `--seed 43`:

%s %s""" % (seeds[0], P, _run_parallel(interface)))
    return None


def _job_seed_text(number):
    seed = _job_seed(number)
    return 'seed %s' % (seed if seed is not None else 41 + number)


def _job_events(number):
    """Where job `number` of the parallel run put its events."""

    job = JOBS[number - 1]
    return _shown(_events_in(job) or os.path.join(job, 'events.lhe.gz'))


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
    text = 'The combined cross section is %s.\n\n' % found if found else ''
    run = _latest_run(interface)
    if run and os.path.exists(os.path.join(run, 'events_decayed.lhe.gz')):
        text += ('%s\n\n  %s\n\n%s'
                 % (textwrap.fill(
                        'MadSpin was still on, so each generation job also '
                        'decayed its own events, with the MadSpin of the '
                        'merged gridpack. The decayed events are combined in',
                        width=76),
                    _shown(os.path.join(run, 'events_decayed.lhe.gz')),
                    _decayed_result(run)))
    return text


# cluster mode trains one job per subprocess: with a single one there is
# nothing to run in parallel, and the cluster half of the lesson shows nothing
MIN_SUBPROCESSES = 2
# what the text says before anything is generated (`tutorial index`, ...)
DEFAULT_SUBPROCESSES = ['g g > t t~', 'Q Qx > t t~']
# subprocesses listed one per line after `generate`; more are counted
LISTED = 8
NUMBERS = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
           7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}


def _cores(amplitude):
    """The production amplitudes: a decay chain's own, or the amplitude."""

    try:
        # a plain Amplitude has no 'decay_chains' and raises here
        if amplitude.get('decay_chains'):
            return amplitude.get('amplitudes')
    except Exception:
        pass
    return [amplitude]


def _beam_swap_key(name):
    """`Qx g > z Qx` and `g Qx > z Qx` are one subprocess, the mirror of the
    other: the same key."""

    initial, _, final = name.partition(' > ')
    return tuple(sorted(initial.split())), final


def _subprocess_tree(interface):
    """[(base, [crossings])] for the last `generate`, as MG5 names them.

    One base per amplitude; a decay chain counts its production processes,
    which are what gets integrated.  With crossing symmetry on (the
    default), a subprocess that is a crossing of another is not generated: it
    is recorded on its base (`crossed_processes`), and each still becomes a
    subprocess of the output -- one training job in cluster mode -- except a
    beam swap, which folds into its mirror.  Empty when nothing is generated.
    """

    tree, seen = [], set()
    for amplitude in getattr(interface, '_curr_amps', None) or []:
        for core in _cores(amplitude):
            try:
                base = core.get('process').base_string()
            except Exception:
                continue
            if _beam_swap_key(base) in seen:
                continue
            seen.add(_beam_swap_key(base))
            crossings = []
            try:
                recorded = core.get('crossed_processes') or []
            except Exception:
                recorded = []
            for entry in recorded:
                try:
                    name = entry[0].base_string()
                except Exception:
                    continue
                if _beam_swap_key(name) not in seen:
                    seen.add(_beam_swap_key(name))
                    crossings.append(name)
            tree.append((base, crossings))
    return tree


def _subprocesses(interface):
    """Every subprocess of the last `generate`, crossings included."""

    found = []
    for base, crossings in _subprocess_tree(interface):
        found.append(base)
        found.extend(crossings)
    return found


def _crossing_note(interface):
    """Which subprocesses are crossings, when crossing symmetry made any."""

    crossed = [(base, crossings)
               for base, crossings in _subprocess_tree(interface) if crossings]
    if not crossed:
        return ''
    lines = '\n'.join('  %s  from `%s`' % (', '.join('`%s`' % c for c in cs),
                                           base)
                      for base, cs in crossed)
    return ("""

Crossing symmetry is on, so some of these are crossings of another:

%s

A crossing has the same diagrams as its base, with legs moved between the
initial and the final state. MG7 generates the diagrams once and evaluates
the crossing with the code of its base. It is still a subprocess of its own,
with its own PDFs and phase space, so it gets its own training job in
cluster mode.""" % lines)


def _merged_quark_note(interface, names):
    """What `Q` and `Qx` mean, when the reader's subprocesses show them."""

    if not {'Q', 'Qx'} & set(' '.join(names).split()):
        return ''
    try:
        model = interface._curr_model
        names = [model.get_particle(pdg).get('name')
                 for pdg in model.get('merged_particles')[81]]
        flavours = ' and '.join([', '.join(names[:-1]), names[-1]]
                                if len(names) > 1 else names)
    except Exception:
        flavours = 'the light quarks'
    return ('\n\n`Q` stands for %s, and `Qx` for their antiquarks.\n'
            'They have the same matrix element, so they share one '
            'subprocess.' % flavours)


def _number(count):
    return NUMBERS.get(count, str(count))


def _listed(names, limit=4):
    """`a`, `b` and `c` -- the first few, and how many more."""

    shown = ['`%s`' % name for name in names[:limit]]
    if len(names) > limit:
        shown.append('%d more' % (len(names) - limit))
    if len(shown) == 1:
        return shown[0]
    return '%s and %s' % (', '.join(shown[:-1]), shown[-1])


def _reader_subprocesses(interface):
    """The reader's subprocesses, or the ones the text assumes."""

    return _subprocesses(interface) or DEFAULT_SUBPROCESSES


def _enough_subprocesses(interface=None, line=None):
    """The gate on the `generate` lesson: at least two subprocesses.

    Any process makes a gridpack, but the cluster half of the tutorial trains
    one job per subprocess, and with a single one there is nothing to run in
    parallel.  So a one-subprocess process does not move the tutorial on: it
    says why and suggests one that does.
    """

    found = _subprocesses(interface)
    if len(found) >= MIN_SUBPROCESSES:
        return True
    what = ('a single subprocess, %s' % _listed(found) if found
            else 'no subprocess')
    return ("""
%(counts)sThat is %(what)s.

Cluster mode, later in this tutorial, gives each subprocess a training job of
its own, so it needs at least two to have anything to run in parallel. The
tutorial stays here.

Generate a simple process with two subprocesses or more, for example:

%(p)s generate p p > t t~""" % {'p': P, 'what': what,
                                 'counts': counts_line(interface)})


def _remember_subprocesses(interface):
    """The `generate` lesson: the reader's own subprocesses, named."""

    names = _reader_subprocesses(interface)
    shown = names if len(names) <= LISTED else names[:LISTED - 1]
    lines = '\n'.join('  `%s`' % name for name in shown)
    if len(shown) < len(names):
        lines += '\n  ... and %d more' % (len(names) - len(shown))
    return ('Remember the %s subprocesses. They matter later, for cluster '
            'mode.\n\n%s%s%s' % (_number(len(names)), lines,
                                 _crossing_note(interface),
                                 _merged_quark_note(interface, names)))


def _training_jobs(interface):
    """The sentence on the training jobs of the reader's process.

    Starts a line of its own and is wrapped here: the names are the reader's,
    so its length is not known when the lesson is written.
    """

    names = _reader_subprocesses(interface)
    if len(names) == 2:
        text = ('Here that means two jobs, one for `%s` and one for `%s`.'
                % tuple(names))
    else:
        text = ('Here that means %s jobs, one for each subprocess listed '
                'after `generate`.' % _number(len(names)))
    return textwrap.fill(text + ' Further jobs then generate the events.',
                         width=76)


FIRST_LAUNCH_HINT = """
The gridpack settings are in the [gridpack] section of the run card, hidden
by default. `update gridpack` shows it, `update list` lists the hidden
sections and `update all` shows them all. `set` also works on a hidden
option directly. To turn the gridpack on, type this, then press Enter.

  set save_gridpack true

The run then optimises the grids as usual and stores the gridpack,
**without generating events**. Optional: `help run_mode` for the other ways
it can end.
"""

RUN_MODE_ADVICE = """Which `run_mode` to pick:

  minimal         The default, right for most samples. Each gridpack run
                  finds the maximum weights itself.
  fix_max_weight  When each job makes few events, a few thousand or less:
                  the maximum weights are found once, here, and all jobs
                  unweight the same way. Slower to create. It also adds
                  bin/event_stream.py, to stream events into Python.
  regular         When this run should also produce a first sample.

Keep the default for this tutorial. %(next)s"""

TURN_IT_ON = "Type `set save_gridpack true`, then press Enter."
START = "Press Enter to start the run."


def _save_gridpack_on():
    """`save_gridpack` in the card being edited right now: True or False, or
    None when the question asked is not one with that parameter."""

    question = extended_cmd.active_question
    run_card = getattr(question, 'run_card', None)
    try:
        return bool(run_card['gridpack.save_gridpack'])
    except Exception:
        return None


def _next_move():
    """What to do next at the card question, given the card as it is."""

    if _save_gridpack_on():
        return "`save_gridpack` is true. " + START
    return TURN_IT_ON


def _first_launch_progress(interface, line):
    words = (line or '').lower().replace('=', ' ').split()
    if words[:2] in (['help', 'run_mode'], ['help', 'gridpack.run_mode']):
        return RUN_MODE_ADVICE % {'next': _next_move()}
    if words[:2] in (['set', 'run_mode'], ['set', 'gridpack.run_mode']):
        return "`run_mode` is set. " + _next_move()
    if words[:2] in (['set', 'save_gridpack'],
                     ['set', 'gridpack.save_gridpack']):
        if _save_gridpack_on() is False:
            return "`save_gridpack` is false: the run would make no gridpack. " \
                   + TURN_IT_ON
        return "Now press Enter to start the run."
    if words[:2] in (['update', 'gridpack'], ['update', 'all'],
                     ['update', 'to_full']):
        return "The [gridpack] section is in the card now. " + _next_move()
    return None


def _first_launch_gate(interface, line):
    """The run this lesson starts has to make the gridpack the next lessons
    are about: Enter, done or 0 wait until `save_gridpack` is true."""

    if _save_gridpack_on() is False:
        return ("Not yet: `save_gridpack` is false, so this run would make no "
                "gridpack. " + TURN_IT_ON)
    return None


# the decays the lesson suggests, when the reader's process has top quarks
TOP_DECAYS = ('decay t > w+ b, w+ > l+ vl', 'decay t~ > w- b~, w- > l- vl~')


def _has_tops(interface):
    """Whether the reader's process has top quarks to decay (True when
    nothing is generated: the text then assumes the suggested p p > t t~)."""

    names = _subprocesses(interface)
    if not names:
        return True
    return any(particle in ('t', 't~')
               for name in names for particle in name.split(' > ')[-1].split())


def _decay_lines(interface=None):
    """The `decay` lines to type at the card question, with what they do."""

    if _has_tops(interface):
        return ('  %-34s the top quark: W+ b, then W+ to e+ or mu+\n'
                '  %-34s the same for the top antiquark'
                % TOP_DECAYS)
    return ('  decay X > ...                      one line per particle of your\n'
            '                                     process to decay, e.g.\n'
            '                                     `decay z > l+ l-`')


MADSPIN_LAUNCH_HINT = """
**Decays in the gridpack.** MadSpin decays the unstable particles of each
event, keeping their spin correlations. When MadSpin is on while the gridpack
is made, everything it needs is prepared now, once, and every run of the
gridpack decays its own events. Where it runs, the gridpack needs neither
MadGraph nor a compiler.

Type these lines, then press Enter.

  madspin=ON                         decay the events with MadSpin
%(decays)s
  set save_gridpack true             make a gridpack, if it is not on any more

The decays are the `decay` lines of `madspin_card.dat`. A `decay` line typed
here replaces the decay of that particle in the card. `l+` stands for e+ and
mu+, `vl` for their neutrinos. To see or edit the whole card, open it from the
list above.
"""

MADSPIN_ON = "Type `madspin=ON`, then press Enter."


def _question_switch(name):
    """The value of switch `name` of the question asked right now, upper
    case, or None when that question has no such switch."""

    question = extended_cmd.active_question
    try:
        return str(question.switch[name]).upper()
    except Exception:
        return None


def _madspin_unavailable():
    """Whether the question asked right now shows MadSpin as not available."""

    return (_question_switch('madspin') or '').startswith('NOT AVAIL')


def _madspin_on():
    """The madspin switch of the question asked right now: True or False, or
    None when the question asked has no such switch, or shows MadSpin as not
    available (nothing to switch on: the next lesson says so)."""

    value = _question_switch('madspin')
    if value is None or _madspin_unavailable():
        return None
    return value not in ('OFF', 'NONE', '')


def _madspin_next_move():
    """What is still missing at the MadSpin card question."""

    if _madspin_on() is False:
        return MADSPIN_ON
    if _save_gridpack_on() is False:
        return TURN_IT_ON
    return "MadSpin and `save_gridpack` are on. " + START


def _madspin_launch_progress(interface, line):
    words = (line or '').replace('=', ' ').split()
    lowered = [word.lower() for word in words]
    if lowered[:1] == ['madspin'] or lowered[:2] == ['set', 'madspin']:
        if _madspin_unavailable():
            return ("MadSpin is not available in this installation, so this "
                    "gridpack cannot decay its events.")
        if _madspin_on():
            return ("MadSpin is on: its card, `madspin_card.dat`, is in the "
                    "list above. Now the decays, or " + _madspin_next_move())
        return "MadSpin is off. " + MADSPIN_ON
    if lowered[:1] == ['decay']:
        return ("`madspin_card.dat` now decays %s that way. %s"
                % (words[1] if len(words) > 1 else 'that particle',
                   _madspin_next_move()))
    if lowered[:2] in (['set', 'save_gridpack'],
                       ['set', 'gridpack.save_gridpack']):
        if _save_gridpack_on() is False:
            return ("`save_gridpack` is false: the run would make no gridpack. "
                    + TURN_IT_ON)
        return "`save_gridpack` is true. " + _madspin_next_move()
    return None


def _madspin_launch_gate(interface, line):
    """The run this lesson starts has to make a gridpack that decays its
    events: Enter, done or 0 wait until MadSpin and `save_gridpack` are on."""

    if _madspin_on() is False:
        return ("Not yet: MadSpin is off, so the gridpack would not decay its "
                "events. " + MADSPIN_ON)
    if _save_gridpack_on() is False:
        return ("Not yet: `save_gridpack` is false, so this run would make no "
                "gridpack. " + TURN_IT_ON)
    return None


def _decayed_result(directory):
    """The sigma x BR of the decayed events in `directory`, as a sentence, or
    '' when it cannot be read."""

    import gzip
    path = os.path.join(directory or '', 'events_decayed.lhe.gz')
    try:
        # line by line to the <init> block: the banner before it carries every
        # card, a large param card included
        init = None
        with gzip.open(path, 'rt') as f:
            for line in f:
                if init is None:
                    if line.strip().startswith('<init>'):
                        init = [line]
                elif line.strip().startswith('</init>'):
                    break
                else:
                    init.append(line)
        return textwrap.fill(
            'Their cross section, in the <init> block, is %.4g pb: the '
            'production cross section times the branching ratio of the '
            'decays.' % float(init[2].split()[0]), width=76) + '\n\n'
    except Exception:
        return ''


def _gridpack_has_madspin(interface):
    """Whether the gridpack of the newest run carries a prepared MadSpin; in
    the archive when `compress` packed it."""

    run = _latest_run(interface)
    if run is None:
        return False
    card = os.path.join('madspin', 'ms_dir', 'madspin_card.dat')
    if os.path.exists(os.path.join(run, 'gridpack', card)):
        return True
    archive = os.path.join(run, 'gridpack.tar.gz')
    if os.path.exists(archive):
        import tarfile
        try:
            with tarfile.open(archive) as tar:
                return any(member.name.endswith(card) for member in tar)
        except Exception:
            return True
    return False


def _madspin_gridpack_made(interface, line=None):
    """Gate: the run made a gridpack that decays its events."""

    refusal = _gridpack_made(interface)
    if refusal:
        return refusal
    if _gridpack_has_madspin(interface):
        return None
    return ("""
That gridpack carries no MadSpin (it has no madspin/ms_dir), so it would not
decay its events. MadSpin was off at the card question, or is not available
in this installation. The tutorial stays here. Launch again, with
`madspin=ON` and the decay lines:

%s %s""" % (P, _launch(interface)))


def _decayed_ran(interface, line=None):
    """Gate: the run of the MadSpin gridpack wrote decayed events."""

    found = _gridpack_events_found(interface)
    if found is None:
        return _gridpack_ran(interface)
    if os.path.basename(found) == 'events_decayed.lhe.gz':
        return None
    return ("""
That run wrote undecayed events, %s, so the tutorial stays here. With
`--no_madspin` a gridpack skips its decays. Run it without that option:

%s %s""" % (found, P, _run_gridpack(interface)))


def _gridpack_run_dir(interface=None):
    """The newest run directory of the gridpack, or None."""

    events = _gridpack_events(interface)
    return os.path.dirname(events) if events else None


CLUSTER_LAUNCH_HINT = """
**Cluster mode.** A normal run trains the subprocesses one after the other,
in one process. Cluster mode gives each subprocess its own training job, and
all of them run in parallel.
%(jobs)s

A real cluster needs a scheduler. This tutorial uses `local_parallel`
instead. It runs the same jobs as separate processes on this machine.
Moving to a real cluster later is one line in the card.

Type these lines, then press Enter.

  set cluster enable                 turn on cluster mode
  set cluster.type local_parallel    run the jobs on this machine
  set events %(events)d                   events in total
  set cluster.events_per_job %(per_job)d    %(gen_jobs)s generation jobs

The jobs share the %(cores)d cores of this machine. `cluster.parallel_jobs`
is how many run at once, each on its share of the cores. The default, -1,
runs one job per core, each on a single core. This run never has more than
%(at_once)d jobs at once (%(train)d training, %(gen)d generation), so
`set cluster.parallel_jobs %(at_once)d` gives each of them %(share)s.
"""

# the events of the cluster run, and how they are split into jobs
CLUSTER_EVENTS = 10000
EVENTS_PER_JOB = 2500


def _cluster_launch_hint(interface=None):
    """The card question of the cluster launch: what cluster mode does with
    the reader's own subprocesses, then the lines to type."""

    train = len(_reader_subprocesses(interface))
    gen = -(-CLUSTER_EVENTS // EVENTS_PER_JOB)
    at_once = min(_cores_here(), max(train, gen))
    return CLUSTER_LAUNCH_HINT % {
        'jobs': _training_jobs(interface), 'events': CLUSTER_EVENTS,
        'per_job': EVENTS_PER_JOB, 'gen_jobs': _number(gen), 'gen': gen,
        'train': train, 'cores': _cores_here(), 'at_once': at_once,
        'share': _cores_word(max(1, _cores_here() // at_once))}


def _card_value(name):
    """Parameter `name` (section.key) of the run card being edited right
    now, or None when the question asked has no such card."""

    run_card = getattr(extended_cmd.active_question, 'run_card', None)
    try:
        return run_card[name]
    except Exception:
        return None


def _cluster_launch_progress(interface, line):
    words = (line or '').lower().replace('=', ' ').split()
    if words[:2] == ['set', 'cluster.events_per_job']:
        missing = _cluster_missing()
        if missing:
            return missing
        return "That is everything. Press Enter to start the run."
    return None


def _cluster_missing():
    """What the cluster run of this lesson still needs in the card, or
    None."""

    if _card_value('cluster.enable') is False:
        return ("Not yet: cluster mode is off. Type `set cluster enable`, "
                "then press Enter.")
    if _card_value('cluster.type') == 'auto':
        return ("Not yet: `cluster.type` is auto, which takes the batch "
                "system of your MadGraph configuration. This tutorial runs "
                "the jobs on this machine: type `set cluster.type "
                "local_parallel`, then press Enter.")
    return None


def _cluster_launch_gate(interface, line):
    """The run this lesson starts is a cluster run on this machine: Enter,
    done or 0 wait for cluster mode and a batch system."""

    return _cluster_missing()


def _cluster_ran(interface, line=None):
    """Gate: the cluster run combined its events into a run directory."""

    run = _latest_run(interface)
    if run is not None and os.path.isdir(os.path.join(run, 'cluster')) and \
            _events_in(run):
        return None
    return ("""
That cluster run did not finish%s, so the tutorial stays here. Its output above
says why, and the log of each job is in the cluster/ directory of the run.
Launch again: the run continues where it stopped, once the cause is fixed.

%s %s""" % (' (%s holds no events)' % _shown(run) if run else '', P,
            _launch(interface)))


GRIDPACK_ONLY_HINT = """
Cluster mode is still on. Type these lines, then press Enter.

  set cluster.mode gridpack_only    stop after the training
  set madnis enable                 train MadNIS
  set madnis.train_batches 2000     a short training

The [gridpack] options also apply to the merged gridpack.
"""


def _gridpack_only_progress(interface, line):
    words = (line or '').lower().replace('=', ' ').split()
    if words[:2] == ['set', 'madnis.train_batches']:
        return _gridpack_only_missing() or \
            "That is everything. Press Enter to start the run."
    return None


def _gridpack_only_missing():
    """What the training-only run of this lesson still needs in the card, or
    None."""

    if _card_value('cluster.enable') is False:
        return ("Not yet: cluster mode is off. Type `set cluster enable`, "
                "then press Enter.")
    if _card_value('cluster.mode') not in (None, 'gridpack_only'):
        return ("Not yet: `cluster.mode` is %s, so the run would generate "
                "events, not stop after the training. Type `set cluster.mode "
                "gridpack_only` (with an underscore), then press Enter."
                % _card_value('cluster.mode'))
    if _card_value('madnis.enable') is not None and \
            _card_value('madnis.enable') is not True:
        return ("Not yet: MadNIS is not on. Type `set madnis enable`, then "
                "press Enter.")
    return None


def _gridpack_only_gate(interface, line):
    """Enter, done or 0 wait until the run stops after the MadNIS
    training."""

    return _gridpack_only_missing()


def _merged_gridpack_made(interface, line=None):
    """Gate: the training-only run left its merged gridpack."""

    run = _latest_run(interface)
    if run is not None and os.path.isdir(os.path.join(run, 'cluster')) and any(
            os.path.exists(os.path.join(run, name))
            for name in ('gridpack', 'gridpack.tar.gz')):
        return None
    return ("""
That run left no merged gridpack%s, so the tutorial stays here. Its output
above says why. Launch again, with `cluster.mode` set to gridpack_only:

%s %s""" % (' in %s' % _shown(run) if run else '', P, _launch(interface)))


def _merged_decay_note(interface):
    """Whether the merged gridpack decayed the events it just made."""

    found = _gridpack_events_found(interface) or ''
    if os.path.basename(found) == 'events_decayed.lhe.gz':
        return (' MadSpin is still on, so the merged gridpack\ncarries it '
                'too and has decayed them.')
    return (' MadSpin was off for this run, so the merged\ngridpack does not '
            'decay them.')


tutorial = Tutorial(
    name='gridpack',
    title='gridpacks and cluster runs',
    description='create and run a gridpack, decay its events with MadSpin, '
                'and parallelise a run with cluster mode',
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

Please start by generating a simple process, so that every run is quick. It
needs at least two subprocesses, because cluster mode gives each subprocess a
training job of its own. For example:

%(p)s generate p p > t t~
""" % {'p': P},
     title='welcome',
     solution='generate p p > t t~'),

Step('generate', lambda interface: """
%(counts)s%(subprocesses)s

%(p)s output %(run)s
""" % {'p': P, 'run': RUN, 'counts': counts_line(interface),
       'subprocesses': _remember_subprocesses(interface)},
     title='a process with several subprocesses',
     # a single subprocess leaves cluster mode nothing to run in parallel:
     # the reader is asked for another process and the tutorial waits here
     gate=_enough_subprocesses,
     solution='output %s' % RUN),

Step('output', lambda interface: """
Whether a run makes a gridpack is a setting of the run card. After the
`launch` command, MG7 offers you the cards to edit. Turn the gridpack on
there. The card question shows what to type.

%(p)s %(launch)s
""" % {'p': P, 'launch': _launch(interface)},
     title='turn the gridpack on',
     question_hint=FIRST_LAUNCH_HINT,
     question_progress=_first_launch_progress,
     question_gate=_first_launch_gate,
     hint="`set save_gridpack true` at the card question, then Enter.",
     solution=_launch),

Step('launch', lambda interface: """
%(found)s
It is self-contained.

  bin/generate_events    the event generator
  Cards/                 grid_run_card.toml, run_card.toml, param_card.dat
  data/                  the optimised channels
  lib/                   the compiled matrix elements
  madspace/              a compiled copy of madspace
  Events/                the output of each gridpack run

The cuts, beams, masses and the integration are fixed in the gridpack. What
can still change, like the number of events, the seed, the devices and the
output format, is read by each run from

  %(grid_card)s

Its defaults are those of the run that made the gridpack, so nothing needs
editing. To change a setting, either edit that card, or give the matching
command-line option. The option is recommended: it applies to that run only,
leaves the card as it is, and lets jobs sharing the gridpack differ.

Now run it. `shell` runs a command from this prompt. In a real study, a job
script would call the gridpack on a cluster node.

%(p)s %(run_gp)s
""" % {'p': P, 'found': _gridpack_found(interface),
       'run_gp': _run_gridpack(interface),
       'grid_card': _grid_run_card(interface)},
     title='what is in a gridpack',
     gate=_gridpack_made,
     hint="`shell PATH/bin/generate_events --events N --seed S` runs the "
          "gridpack.",
     solution=_run_gridpack),

Step('shell', lambda interface: """
That run had no survey and no training, so it was much faster than a full
run. Its last line says where the events are:

  %(events)s

A gridpack writes LHE by default, since its events usually go to a shower.
`--output_format` selects one of the npy formats instead.

Four options matter when several jobs share a gridpack.

  --seed                  must differ between jobs. The default, -1, draws a
                          new seed and records it.
  --output_dir            writes the events there instead of into the
                          gridpack.
  --temp_output_dir       holds the large temporary files. `auto` makes a new
                          directory per run in $TMPDIR, so jobs do not collide.
  --cpu_thread_pool_size  how many cores a job uses. The default, -1, takes
                          them all.

`--help` lists all options.

Try it: run two jobs at the same time, each with its own seed and output
directory, and each on half of the %(cores)d cores of this machine. Each `&`
starts a job in the background, and `wait` waits for both. Their output goes
to JOB_1.log and JOB_2.log.

%(p)s %(parallel)s
""" % {'p': P, 'events': _gridpack_events(interface),
       'cores': _cores_here(), 'parallel': _run_parallel(interface)},
     title='running a gridpack',
     gate=_gridpack_ran,
     hint="Two `generate_events` commands joined by `&`, each with its own "
          "`--seed` and `--output_dir`, then `& wait` (no `;`: MG7 splits "
          "the line there).",
     solution=_run_parallel),

Step('shell', lambda interface: """
Both jobs ran at the same time, on %(half)d cores each. Each one wrote its
events into its own directory, and the last line of its log says where:

  %(job1)s      %(seed1)s
  %(job2)s      %(seed2)s

Different seeds give independent samples, so the two files together are one
sample twice as large. The same seed twice would give the same events twice.
On a cluster, each job is one such command, on a node of its own.

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

A gridpack can also **decay its events**. Launch the same output again, with
MadSpin on: the card question explains it and lists the lines to type.

%(p)s %(launch)s
""" % {'p': P, 'half': _half_cores(), 'job1': _job_events(1),
       'job2': _job_events(2), 'seed1': _job_seed_text(1),
       'seed2': _job_seed_text(2),
       'launch': _launch(interface)},
     title='parallel gridpack jobs, and what to put in a gridpack',
     gate=_jobs_ran,
     question_hint=lambda interface: MADSPIN_LAUNCH_HINT % {
         'decays': _decay_lines(interface)},
     question_progress=_madspin_launch_progress,
     question_gate=_madspin_launch_gate,
     hint="At the card question, `madspin=ON`, the `decay` lines and "
          "`set save_gridpack true`, then Enter.",
     solution=_launch),

Step('launch', lambda interface: """
%(found)s
It is the gridpack of before, plus MadSpin, prepared for your decays:

  madspin/ms_dir/          the prepared decays: one small gridpack for each
                           decay channel, the compiled matrix elements of the
                           spin correlations and the maximum weights
  madspin/python/          the MadGraph and MadSpin Python it runs with, and
                           the model
  Cards/madspin_card.dat   the decays, for the record

All of it was built during this run, from a sample the new gridpack made
itself. So the decays cost each gridpack run seconds, and the gridpack still
runs without MadGraph and without a compiler.

Run it as before.

%(p)s %(run_gp)s
""" % {'p': P, 'found': _gridpack_found(interface),
       'run_gp': _run_gridpack(interface)},
     title='decays in the gridpack',
     gate=_madspin_gridpack_made,
     hint="`shell PATH/bin/generate_events --events N --seed S` runs the "
          "gridpack.",
     solution=_run_gridpack),

Step('shell', lambda interface: """
That run generated the events, then decayed them with MadSpin, in one go.
Its last line says where the decayed events are:

  %(events)s

%(xsec)sMadSpin draws its own random numbers, from a seed derived from `--seed`.
The same `--seed` gives the same decayed events, so jobs sharing the gridpack
only need different seeds, as before.

  --no_madspin       write the undecayed events, as without MadSpin
  --keep_undecayed   keep the undecayed events.lhe.gz next to the decayed ones

The decays themselves are fixed in the gridpack, like the cuts. To change
them, make a new gridpack. MadSpin decays LHE events, so the npy output
formats need `--no_madspin`.

Next comes **cluster mode**, which splits a whole run into jobs that run in
parallel. Launch the same output again: the card question explains it and
lists the lines to type.

%(p)s %(launch)s
""" % {'p': P, 'events': _gridpack_events(interface),
       'xsec': _decayed_result(_gridpack_run_dir(interface)),
       'launch': _launch(interface)},
     title='running a gridpack with decays',
     gate=_decayed_ran,
     question_hint=_cluster_launch_hint,
     question_progress=_cluster_launch_progress,
     question_gate=_cluster_launch_gate,
     hint="At the card question, `set cluster enable` and "
          "`set cluster.type local_parallel`, then Enter.",
     solution=_launch),

Step('launch', lambda interface: """
%(result)sThat was a whole run, split into jobs.

  1. **Training.** One job per subprocess, here %(nsub)s in parallel. Each
     job runs its own survey and training and saves a gridpack.
  2. **Merging.** The %(nsub)s gridpacks become one with all channels.
  3. **Generation.** Several jobs run the merged gridpack, each with its own
     seed. `events_per_job` sets how many.
  4. **Combining.** The events are joined into an ordinary run in
     `%(run)s`, in the `output_format` of the run card. The cross section is
     averaged over the jobs. With MadSpin on, each generation job decays its
     own events first, and the decayed events are the ones joined.

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
       'nsub': _number(len(_reader_subprocesses(interface))),
       'run': _shown(_latest_run(interface) or
                     os.path.join(output_name(interface, RUN), 'Events',
                                  'run_02'))},
     title='a cluster run, locally',
     gate=_cluster_ran,
     question_hint=GRIDPACK_ONLY_HINT,
     question_progress=_gridpack_only_progress,
     question_gate=_gridpack_only_gate,
     hint="At the card question, `set cluster.mode gridpack_only`, "
          "`set madnis enable` and `set madnis.train_batches 2000`, "
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
     gate=_merged_gridpack_made,
     hint="`shell PATH/bin/generate_events --events N --seed S` runs the "
          "gridpack.",
     solution=_run_gridpack),

Step('shell', lambda interface: """
These events come from the merged gridpack, with the trained MadNIS
channels.%(decayed)s They are in

  %(events)s

Compare `unw. eff.` in the output with the first gridpack run. The MadNIS
channels usually unweight better, which is what makes a long training pay
off for a large sample. On a cluster, each job would run this command with
its own `--seed`.

Cluster mode stays on in the card of this output. Turn it off with
`set cluster disable` at the next launch. MadSpin stays on as well: in
cluster mode each generation job decays its own events with the MadSpin of
the merged gridpack. The other tools -- the shower, Delphes, reweighting --
run on the events of a cluster run afterwards.

In short:

  * `save_gridpack = true` turns a run into a gridpack. `[gridpack]` sets
    what goes into it and how the run ends.
  * `bin/generate_events --events N --seed S` runs a gridpack.
  * MadSpin on while the gridpack is made: every gridpack run decays its own
    events. `--no_madspin` skips it, `--keep_undecayed` keeps both.
  * `[cluster] enable = true` splits a run into jobs. `type` picks the
    scheduler, or `local_parallel` for this machine. `mode` picks events or
    a merged gridpack.

The documentation lists every option.

  %(docs)s/gridpacks.html       gridpacks
  %(docs)s/cluster.html         cluster mode, bin/merge_gridpacks
  %(docs)s/event_stream.html    streaming events into Python

%(see_also)s

Leave with `tutorial stop`.
""" % {'events': _gridpack_events(interface), 'docs': DOCS,
       'decayed': _merged_decay_note(interface),
       'see_also': tutorials.where_next()},
     title='running the merged gridpack',
     gate=_gridpack_ran),

    ],
)
