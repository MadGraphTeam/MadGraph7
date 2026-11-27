################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
################################################################################
"""MadSpin from a prepared ms_dir: the decay step of an mg7 gridpack.

An mg7 gridpack that decays its events carries a *prepared* ms_dir: the decay
channels as mg7 gridpacks, the compiled density matrix elements and the maximum
weights, all built once by ``prepare`` when the gridpack is made. ``decay``
then only runs it, without a MadGraph installation and without a compiler
(the Python of MadGraph, MadSpin and the UFO model is shipped with the
gridpack, see :func:`install_runtime`)::

    python -m MadSpin.gridpack_decay prepare --card CARD --events EVENTS \\
        --ms_dir DIR [--seed N] [--nb_core N]
    python -m MadSpin.gridpack_decay decay --ms_dir DIR --events EVENTS \\
        --seed N [--output OUT] [--nb_core N]

Both run MadSpin in this process, so each call has to be a process of its own:
MadSpin seeds Python's random generator only once per process.

The ms_dir is never written by ``decay``: every run works in a shadow of it
(:func:`shadow_ms_dir`), where what MadSpin writes -- the decay pools of this
run, the cards it refreshes -- are files of its own and everything else links
back to the prepared directory. So the jobs of a cluster run can decay their
events with one gridpack at the same time, and the gridpack may be read-only.
"""

import argparse
import glob
import gzip
import logging
import os
import shutil
import sys
import tempfile

# the madspin card of the prepared ms_dir, what decay runs
CARD_NAME = 'madspin_card.dat'

# the largest seed MadSpin accepts (ranmar)
MAX_SEED = 30081 * 30081

logger = logging.getLogger('decay')


class GridpackDecayError(Exception):
    """MadSpin cannot be prepared for, or run from, a gridpack."""


# ---------------------------------------------------------------------------
# the madspin card
# ---------------------------------------------------------------------------
def gridpack_card(text):
    """The madspin card as a gridpack ships it: the decays and the settings,
    without the lines the caller provides at each run (the event file, the
    ms_dir, the seed) nor the launch.

    Raises GridpackDecayError on what a gridpack cannot run: an ``import
    model`` line (it names a model and a param_card by paths of the machine
    that made the gridpack), the legacy spin modes, and madevent decay
    pools."""
    kept = []
    for line in text.splitlines():
        stripped = line.split('#', 1)[0].strip()
        words = stripped.lower().split()
        if not words:
            kept.append(line)
            continue
        if words[0] in ('launch', 'exit', 'quit'):
            continue
        if words[0] == 'import':
            if len(words) > 1 and words[1] == 'model':
                raise GridpackDecayError(
                    "the madspin card imports a model ('%s'): a gridpack runs "
                    "MadSpin with the model and the parameters of its events "
                    "only" % stripped)
            continue
        if words[0] == 'set' and len(words) > 1:
            if words[1] in ('ms_dir', 'seed', 'use_old_dir'):
                continue
            if words[1] == 'spinmode' and len(words) > 2 and \
                    words[2] in ('madspin_v1', 'onshell_v1', 'bridge'):
                raise GridpackDecayError(
                    "spinmode %s cannot run from a gridpack: use madspin "
                    "(full), onshell or none" % words[2])
            if words[1] == 'decay_generator' and len(words) > 2 and \
                    words[2] == 'madevent':
                raise GridpackDecayError(
                    "a gridpack generates its decays with mg7 decay gridpacks: "
                    "remove 'set decay_generator madevent' from the madspin "
                    "card")
        kept.append(line)
    return '\n'.join(kept).strip('\n') + '\n'


def run_card(body, events, ms_dir, seed, nb_core=None, prepare=False):
    """The command file of one MadSpin run: the shipped card ``body`` with the
    event file, the ms_dir and the seed of this run."""
    lines = ['import %s' % events, 'set ms_dir %s' % ms_dir]
    if seed:
        lines.append('set seed %d' % int(seed))
    lines.append(body.rstrip('\n'))
    if nb_core:
        # after the card: the caller's cores win over the card's
        lines.append('set nb_core %d' % int(nb_core))
    lines.append('launch --prepare_only' if prepare else 'launch')
    return '\n'.join(lines) + '\n'


def madspin_seed(seed):
    """The seed of the MadSpin step of a gridpack run of seed ``seed`` (any
    integer, a gridpack seed is 64 bit): reproducible, in [1, 30081**2], and
    another stream than the production's."""
    import hashlib
    digest = hashlib.sha256(('%d:madspin' % int(seed)).encode()).digest()
    return int.from_bytes(digest, 'big') % (MAX_SEED - 1) + 1


# ---------------------------------------------------------------------------
# running MadSpin
# ---------------------------------------------------------------------------
def _run_madspin(command_file):
    """Run the MadSpin command file in this process, from its directory (the
    paths of the file are relative to it, so that the banner MadSpin writes
    names no directory of this run); raise on failure."""
    import MadSpin.interface_madspin as interface_madspin
    previous = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(command_file)))
    try:
        return _run_madspin_here(os.path.basename(command_file),
                                 interface_madspin)
    finally:
        os.chdir(previous)


def _run_madspin_here(command_file, interface_madspin):
    cmd = interface_madspin.MadSpinInterface()
    cmd.use_rawinput = False
    cmd.haspiping = False
    try:
        # exec_cmd, without the error handling of the command loop: a failing
        # line raises here rather than being reported and skipped
        cmd.import_command_file(command_file)
    finally:
        try:
            cmd.do_quit('')
        except BaseException:
            pass
    return cmd


def _plain_copy(events, target):
    """Copy the LHE file ``events`` (gzipped or not) to ``target`` as plain
    text. MadSpin decompresses its input in place: it works on a copy."""
    opener = gzip.open if events.endswith('.gz') else open
    with opener(events, 'rb') as fin, open(target, 'wb') as fout:
        shutil.copyfileobj(fin, fout, 4 * 1024 * 1024)


def clean_prepared(ms_dir):
    """Drop what the prepare run left that a later run does not reuse: the
    decay pools it drew from and its bookkeeping. A gridpack ships the
    rest."""
    for pattern in (os.path.join('decay_*', 'Events', '*'),
                    os.path.join('decay_*', 'Events', '.*'),
                    'ms_gen_*.json', 'ms_refill*', 'ms_status*', 'ms_wstatus*',
                    'seeds.dat'):
        for path in glob.glob(os.path.join(ms_dir, pattern)):
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except OSError:
                    pass


def prepare(card, events, ms_dir, seed=None, nb_core=None):
    """Prepare ``ms_dir`` for decays run from a gridpack: build the decay
    gridpacks, the density matrix elements and the maximum weights from the
    production events ``events`` with the madspin card ``card``, without
    decaying them. The card is kept in ``ms_dir`` as the gridpack runs it."""
    with open(card) as fsock:
        body = gridpack_card(fsock.read())
    ms_dir = os.path.abspath(ms_dir)
    os.makedirs(ms_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix='madspin_prepare_', dir=os.path.dirname(ms_dir))
    try:
        plain = os.path.join(work, 'events.lhe')
        _plain_copy(events, plain)
        command_file = os.path.join(work, 'madspin_card.dat')
        with open(command_file, 'w') as fsock:
            fsock.write(run_card(body, 'events.lhe', ms_dir, seed, nb_core,
                                 prepare=True))
        _run_madspin(command_file)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    clean_prepared(ms_dir)
    with open(os.path.join(ms_dir, CARD_NAME), 'w') as fsock:
        fsock.write(body)
    return ms_dir


def shadow_ms_dir(source, target):
    """A directory ``target`` that MadSpin can run in as if it were the
    prepared ``source``, without writing in ``source``.

    The top-level files (the caches and stamps, all small) are copied. The
    decay gridpacks and the matrix-element trees are directories of their own
    whose entries link back to ``source``, but for the ones a run writes in:
    the ``Events`` of a decay gridpack (this run's pools) and the ``Cards`` of
    a matrix-element tree (refreshed by every run). Every other directory is a
    link."""
    source = os.path.realpath(source)
    os.makedirs(target)
    for name in sorted(os.listdir(source)):
        src = os.path.join(source, name)
        dst = os.path.join(target, name)
        if not os.path.isdir(src):
            shutil.copy2(src, dst)
            continue
        if not (name.startswith('decay_') or name.startswith('madspin_me')
                or name.startswith('madspin_decay')):
            os.symlink(src, dst)
            continue
        os.mkdir(dst)
        for entry in sorted(os.listdir(src)):
            if entry == 'Events' and name.startswith('decay_'):
                os.mkdir(os.path.join(dst, entry))
            elif entry == 'Cards' and not name.startswith('decay_'):
                shutil.copytree(os.path.join(src, entry), os.path.join(dst, entry))
            elif os.path.isfile(os.path.join(src, entry)) and \
                    entry.endswith(('.dat', '.log')):
                shutil.copy2(os.path.join(src, entry), os.path.join(dst, entry))
            else:
                os.symlink(os.path.join(src, entry), os.path.join(dst, entry))
    return target


def _gzip_to(path, target):
    """``path`` (plain or gzipped) as the gzipped file ``target``."""
    if path.endswith('.gz'):
        shutil.move(path, target)
        return
    with open(path, 'rb') as fin, gzip.open(target, 'wb', compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout, 4 * 1024 * 1024)
    os.remove(path)


def decay(ms_dir, events, seed, output=None, nb_core=None, work_dir=None):
    """Decay the LHE events ``events`` with the prepared ``ms_dir`` and the
    seed ``seed`` (in [1, 30081**2], see madspin_seed); returns the decayed
    file, gzipped, ``output`` or ``<events>_decayed.lhe.gz`` next to the
    input. ``work_dir`` is where the run works (default: next to the
    output)."""
    ms_dir = os.path.realpath(ms_dir)
    card = os.path.join(ms_dir, CARD_NAME)
    if not os.path.exists(card):
        raise GridpackDecayError('%s is not a prepared ms_dir (no %s)'
                                 % (ms_dir, CARD_NAME))
    with open(card) as fsock:
        body = fsock.read()
    events = os.path.abspath(events)
    if output is None:
        base = events[:-3] if events.endswith('.gz') else events
        output = base.replace('.lhe', '_decayed.lhe') + '.gz' \
            if '.lhe' in os.path.basename(base) else base + '_decayed.lhe.gz'
    output = os.path.abspath(output)
    work = tempfile.mkdtemp(prefix='madspin_',
                            dir=work_dir or os.path.dirname(output))
    try:
        shadow = shadow_ms_dir(ms_dir, os.path.join(work, 'ms_dir'))
        plain = os.path.join(work, 'events.lhe')
        _plain_copy(events, plain)
        command_file = os.path.join(work, 'madspin_card.dat')
        with open(command_file, 'w') as fsock:
            fsock.write(run_card(body, 'events.lhe', os.path.basename(shadow),
                                 seed, nb_core))
        _run_madspin(command_file)
        produced = [path for path in (
            os.path.join(work, 'events_decayed.lhe.gz'),
            os.path.join(work, 'events_decayed.lhe')) if os.path.exists(path)]
        if not produced:
            raise GridpackDecayError('MadSpin produced no decayed events in %s'
                                     % work)
        _gzip_to(produced[0], output)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return output


# ---------------------------------------------------------------------------
# the Python a gridpack ships
# ---------------------------------------------------------------------------
# the packages MadSpin imports, from the root of a MadGraph installation
RUNTIME_PACKAGES = ('madgraph', 'MadSpin', 'madmatrix', 'aloha', 'mg5decay',
                    os.path.join('vendor', 'ply'))
# single files, as glob patterns (what the run card and MasterCmd read)
RUNTIME_FILES = ('VERSION', os.path.join('vendor', '__init__.py'),
                 os.path.join('models', '__init__.py'),
                 # MasterCmd checks it at start up
                 os.path.join('Template', 'LO', 'Source', '.make_opts'),
                 os.path.join('Template', 'LO', 'Source', 'banner_header.txt'),
                 # the lepton densities a run card accepts
                 os.path.join('Template', 'Common', 'Source', 'PDF',
                              'lep_densities', '*', 'info'))
_RUNTIME_IGNORE = ('__pycache__', '*.pyc', '*.pyo', 'tests', '*.so', '*.dylib',
                   '*.o', '*.a', 'build', 'install')


def install_runtime(target, model_dir, mg_root=None):
    """Copy what MadSpin needs to run without a MadGraph installation into
    ``target``, a root to put on PYTHONPATH: the Python packages, the
    VERSION and input/ files, and the UFO model ``model_dir`` (as
    ``models/<name>``, where MadSpin looks it up)."""
    if mg_root is None:
        import madgraph
        mg_root = os.path.dirname(os.path.dirname(os.path.abspath(madgraph.__file__)))
    os.makedirs(target, exist_ok=True)
    ignore = shutil.ignore_patterns(*_RUNTIME_IGNORE)
    for name in RUNTIME_PACKAGES:
        shutil.copytree(os.path.join(mg_root, name), os.path.join(target, name),
                        ignore=ignore, dirs_exist_ok=True)
    for pattern in RUNTIME_FILES:
        for path in glob.glob(os.path.join(mg_root, pattern)):
            name = os.path.relpath(path, mg_root)
            os.makedirs(os.path.dirname(os.path.join(target, name)), exist_ok=True)
            shutil.copy2(path, os.path.join(target, name))
    # ... and writes make_opts from it when it is missing: ship it, so that a
    # read-only gridpack does not need the write
    make_opts = os.path.join(target, 'Template', 'LO', 'Source')
    if os.path.exists(os.path.join(make_opts, '.make_opts')):
        shutil.copy2(os.path.join(make_opts, '.make_opts'),
                     os.path.join(make_opts, 'make_opts'))
    if os.path.isdir(os.path.join(mg_root, 'input')):
        shutil.copytree(os.path.join(mg_root, 'input'), os.path.join(target, 'input'),
                        ignore=ignore, dirs_exist_ok=True)
    os.makedirs(os.path.join(target, 'models'), exist_ok=True)
    if model_dir:
        shutil.copytree(model_dir,
                        os.path.join(target, 'models', os.path.basename(
                            os.path.normpath(model_dir))),
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc',
                                                      '*.pkl'),
                        dirs_exist_ok=True)
    return target


# ---------------------------------------------------------------------------
def _setup_logging(level):
    logging.basicConfig(level=level, format='%(message)s', stream=sys.stdout)
    for name in ('madgraph', 'madevent', 'decay', 'cmdprint', 'MadSpin'):
        logging.getLogger(name).setLevel(level)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python -m MadSpin.gridpack_decay', description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare', help='prepare an ms_dir for a gridpack')
    prep.add_argument('--card', required=True)
    prep.add_argument('--events', required=True)
    prep.add_argument('--ms_dir', required=True)
    prep.add_argument('--seed', type=int, default=None)
    prep.add_argument('--nb_core', type=int, default=None)
    run = sub.add_parser('decay', help='decay events with a prepared ms_dir')
    run.add_argument('--ms_dir', required=True)
    run.add_argument('--events', required=True)
    run.add_argument('--seed', type=int, required=True)
    run.add_argument('--output', default=None)
    run.add_argument('--nb_core', type=int, default=None)
    run.add_argument('--work_dir', default=None)
    for each in (prep, run):
        each.add_argument('--log_level', default='INFO')
    args = parser.parse_args(argv)
    _setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    try:
        if args.action == 'prepare':
            prepare(args.card, args.events, args.ms_dir, args.seed, args.nb_core)
        else:
            out = decay(args.ms_dir, args.events, args.seed, args.output,
                        args.nb_core, args.work_dir)
            print('Decayed events written to %s' % out)
    except GridpackDecayError as error:
        sys.exit('MadSpin: %s' % error)
    return 0


if __name__ == '__main__':
    sys.exit(main())
