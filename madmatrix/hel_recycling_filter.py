"""The filter of a --hel_recycling build: the good helicities and the
(helicity, amplitude) pairs that vanish, measured by a probe build of the
plain code at the param_card(s) the matrix element will run at.

The recycled build of a P* directory is generated for one filter: the rows
it builds are the good helicities, and the amplitudes it drops are the ones
that vanish at them. Both depend on the parameters -- a coupling set to zero
kills its diagrams, a helicity can open up with a mass -- so the filter is
measured at generation time (madmatrix/output.py, at the output's own
param_card) and again whenever a run is launched at other parameters
(update(), from madgraph/mg7/madgraph_process.py, and before a MadtRex
reweighting evaluates the libraries at the cards of its launches).

What is kept in the output:
  - P*/HelRecyclingRecipe.json: what the exporter knows of the matrix element
    (model_handling.OneProcessExporterMadMatrix.hr_recipe), to rewrite the
    recycled build without it;
  - P*/HelRecyclingFilter.json: the filter the build in that directory is
    for, and the cards it was measured at (so that a launch at the same cards
    measures nothing, and one whose filter comes out the same rewrites and
    recompiles nothing);
  - P*/HelRecyclingTag.mk: the tag a build for another filter than the
    output's own carries in its library name (see below);
  - SubProcesses/hrprobe/: the probe driver (check_sa) an mg7 output does
    not ship otherwise;
  - .hrprobe/: the probe builds, a mirror of the output, kept so that the
    next card is measured by running them (milliseconds) rather than by
    building them again (the probe is the plain code: it does not depend on
    the filter, and its parameters are read at run time).

Why a tag: a process that has loaded a matrix-element library keeps it --
the dynamic loader identifies a library by its path, and a C++ library with
inline statics cannot even be unloaded on Linux -- so a library rebuilt in
place for the next point of a scan, or for a reweighting after the
generation, would never be seen. A build for any filter but the output's own
is therefore named after its filter (lib..._<P>_hr<hash>_<backend>_<fptype>.so),
and subprocesses.json is updated to it: a name always means one build, the
libraries already loaded stay right for what they were loaded for, and
coming back to a filter built before finds its library.
"""

import glob
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess

pjoin = os.path.join

logger = logging.getLogger('madgraph.madmatrix.hel_recycling')

RECIPE = 'HelRecyclingRecipe.json'
STATE = 'HelRecyclingFilter.json'
TAG_MK = 'HelRecyclingTag.mk'
PROBE_SOURCES = 'hrprobe'     # SubProcesses/hrprobe
PROBE_TREE = '.hrprobe'       # <output>/.hrprobe
PROBE_MAKEFILE = 'hrprobe.mk'
# what a recycled build adds to the P* directory
GENERATED = ('HelRecycling.inc', 'HelRecyclingCalls.inc', 'HelRecyclingJamps.inc')
CHUNKS = 'HelRecyclingChunks_*.cc'
# build products, never copied into the probe tree
_BUILD_ARTEFACTS = ('build.*', '.build.*', '*.o', '*.so', '*.a', '*.d', '*.exe',
                    '.check_sa_backend', '.resolved-backend', '__pycache__')
_TAG_RE = re.compile(r'_hr[0-9a-f]{8}(?=_\{device\})')


# -- cards and filters -------------------------------------------------------

def card_digest(path):
    """A digest of the parameters of a param_card: the values only, so that a
    card rewritten with other comments or another layout is the same card."""
    try:
        from madgraph.models.check_param_card import ParamCard
        card = ParamCard(path)
        items = []
        for name in sorted(card):
            for param in card[name]:
                items.append((name, tuple(param.lhacode), repr(param.value)))
        text = repr(sorted(items))
    except Exception:
        with open(path, 'rb') as fsock:
            text = fsock.read().decode(errors='replace')
    return hashlib.sha1(text.encode()).hexdigest()


def cards_key(cards):
    """The cards a filter is measured at, as a sorted list of their digests."""
    return sorted(set(card_digest(card) for card in cards))


def parse_probe_output(output):
    """The good helicities (1-based rows of cHel) and the (row, amplitude)
    pairs that vanish, as printed by sigmaKin_setGoodHel under MG_HR_PROBE --
    the protocol madevent's helicity filter prints."""
    good, zeros = [], []
    for line in output.splitlines():
        if line.startswith('Matrix Element/Good Helicity:'):
            good.append(int(line.split()[-1]))
        elif line.startswith('HEL/ZEROAMP:'):
            _me, hel, amp = line.split()[1:4]
            zeros.append((int(hel), int(amp)))
    return sorted(set(good)), sorted(set(zeros))


def combine(measurements):
    """The filter that holds at every card: the helicities good at one of them
    at least, and the amplitudes that vanish at every card where their
    helicity is good. (Where a helicity is not good, the run time does not
    evaluate it either: the probe keeps every row above a far lower threshold
    than the run-time scan does.)"""
    good = sorted(set(hel for hels, _ in measurements for hel in hels))
    candidates = set(pair for _, zeros in measurements for pair in zeros)
    sets = [(set(hels), set(map(tuple, zeros))) for hels, zeros in measurements]
    zeros = sorted(pair for pair in candidates
                   if all(pair in zs for hs, zs in sets if pair[0] in hs))
    return good, [list(pair) for pair in zeros]


def signature(state):
    """What the build of a P* directory is: its filter, or the plain code."""
    if not state.get('recycled'):
        return {'recycled': False}
    return {'recycled': True, 'good_hels': list(state['good_hels']),
            'zero_perhel': [list(pair) for pair in state['zero_perhel']]}


def tag_of(state):
    """'' for the output's own build, else _hr<hash of the build>."""
    sig = signature(state)
    if sig == state.get('output'):
        return ''
    return '_hr' + hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:8]


def read_state(pdir):
    try:
        with open(pjoin(pdir, STATE)) as fsock:
            return json.load(fsock)
    except (OSError, ValueError):
        return None


def write_state(pdir, state):
    with open(pjoin(pdir, STATE), 'w') as fsock:
        json.dump(state, fsock, indent=1)


# -- the probe ---------------------------------------------------------------

class ProbeTree(object):
    """<output>/.hrprobe: a mirror of the output with only what a probe build
    of the plain code needs (src, backend, the SubProcesses makefiles and one
    copy of each P* directory measured), so that the probe never touches the
    output's own builds and libraries. The probe builds are kept: measuring
    another card is then one run of each."""

    def __init__(self, root, nb_core=None):
        self.root = os.path.realpath(root)
        self.path = pjoin(self.root, PROBE_TREE)
        self.nb_core = nb_core or os.cpu_count() or 1

    def reset(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def _skeleton(self):
        if os.path.isfile(pjoin(self.path, '.ready')):
            return
        self.reset()
        ignore = shutil.ignore_patterns(*_BUILD_ARTEFACTS)
        for name in ('src', 'backend', 'include'):
            source = pjoin(self.root, name)
            if os.path.isdir(source):
                shutil.copytree(source, pjoin(self.path, name), symlinks=True,
                                ignore=ignore)
        for name in ('lib', 'Cards', 'SubProcesses'):
            os.makedirs(pjoin(self.path, name), exist_ok=True)
        subproc = pjoin(self.root, 'SubProcesses')
        for name in os.listdir(subproc):
            source = pjoin(subproc, name)
            if name == PROBE_SOURCES and os.path.isdir(source):
                shutil.copytree(source, pjoin(self.path, 'SubProcesses', name))
            elif not os.path.isdir(source) or os.path.islink(source):
                shutil.copy2(source, pjoin(self.path, 'SubProcesses', name),
                             follow_symlinks=False)
        open(pjoin(self.path, '.ready'), 'w').close()

    def available(self):
        """Whether a probe can be built here: check_sa.cc is in a standalone
        output, the mg7 output keeps it in SubProcesses/hrprobe."""
        subproc = pjoin(self.root, 'SubProcesses')
        return os.path.isfile(pjoin(subproc, PROBE_SOURCES, PROBE_MAKEFILE)) or \
            os.path.isfile(pjoin(subproc, 'check_sa.cc'))

    def build(self, pdir):
        """Build the probe of P* directory `pdir` (if not built already).
        Returns None, or what went wrong."""
        self._skeleton()
        name = os.path.basename(os.path.normpath(pdir))
        target = pjoin(self.path, 'SubProcesses', name)
        if os.path.isfile(pjoin(target, 'check_sa.exe')):
            return None
        shutil.rmtree(target, ignore_errors=True)
        # the plain code: the probe build ignores the recycled files anyway
        # (MG_HR_PROBE, HRPROBE=1), and it must not carry another's library tag
        shutil.copytree(pjoin(self.root, 'SubProcesses', name), target, symlinks=True,
                        ignore=shutil.ignore_patterns(
                            *(_BUILD_ARTEFACTS + GENERATED + (CHUNKS, RECIPE, STATE, TAG_MK))))
        makeargs = []
        sources = pjoin(self.path, 'SubProcesses', PROBE_SOURCES)
        if os.path.isdir(sources):
            for entry in os.listdir(sources):
                dest = pjoin(target, entry)
                if os.path.lexists(dest):
                    os.remove(dest)
                shutil.copy2(pjoin(sources, entry), dest)
            # the standalone makefile includes madmatrix.mk by name
            if not os.path.lexists(pjoin(target, 'madmatrix.mk')):
                os.symlink(pjoin(os.pardir, 'madmatrix.mk'), pjoin(target, 'madmatrix.mk'))
            makeargs = ['-f', PROBE_MAKEFILE]
        built = subprocess.run(
            ['make'] + makeargs + ['-j%d' % self.nb_core,
                                   'BACKEND=scalar', 'FPTYPE=d', 'HRPROBE=1'],
            cwd=target, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if built.returncode != 0 or not os.path.isfile(pjoin(target, 'check_sa.exe')):
            shutil.rmtree(target, ignore_errors=True)
            return 'the probe build failed:\n%s' % \
                built.stdout.decode(errors='replace')[-1500:]
        # only check_sa.exe and the libraries are used again (the common
        # library's objects are kept: the next directory's probe links it)
        for dirpath, _dirs, names in os.walk(target):
            for name in names:
                if name.endswith(('.o', '.d', '.pch', '.gch')):
                    os.remove(pjoin(dirpath, name))
        return None

    def run(self, pdir, card):
        """(good helicities, vanishing pairs) of the probe of `pdir` at `card`,
        or raise RuntimeError."""
        name = os.path.basename(os.path.normpath(pdir))
        shutil.copy(card, pjoin(self.path, 'Cards', 'param_card.dat'))
        target = pjoin(self.path, 'SubProcesses', name)
        # perf mode: 64 distinct RAMBO events, which the probe's good-helicity
        # scan samples (matrix mode repeats ONE point)
        env = dict(os.environ, MG_DUMP_GOODHEL='1')
        run = subprocess.run([pjoin(target, 'check_sa.exe'), 'perf', '1', '64', '1'],
                             cwd=target, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        output = run.stdout.decode(errors='replace')
        if run.returncode != 0:
            raise RuntimeError('the probe run failed at %s:\n%s' % (card, output[-1500:]))
        good, zeros = parse_probe_output(output)
        if not good:
            raise RuntimeError('the probe found no helicity at %s' % card)
        return good, zeros

    def measure(self, pdir, cards):
        """The filter of `pdir` at all of `cards` (see combine), or raise
        RuntimeError."""
        for attempt in (0, 1):
            error = self.build(pdir)
            if error:
                raise RuntimeError(error)
            try:
                return combine([self.run(pdir, card) for card in cards])
            except RuntimeError:
                if attempt:
                    raise
                # a probe kept from before that no longer runs (the output
                # moved, another compiler): build it again, once
                shutil.rmtree(pjoin(self.path, 'SubProcesses',
                                    os.path.basename(os.path.normpath(pdir))),
                              ignore_errors=True)


# -- writing a build ---------------------------------------------------------

def remove_recycled(pdir):
    """Take the recycled build out of `pdir`: the plain code remains."""
    for name in GENERATED:
        if os.path.exists(pjoin(pdir, name)):
            os.remove(pjoin(pdir, name))
    for path in glob.glob(pjoin(pdir, CHUNKS)):
        os.remove(path)


def write_build(pdir, recipe, good, zeros):
    """Write the recycled build of `pdir` for the filter (good, zeros), or the
    plain code when good is None or the build is declined for its size.
    Returns the number of rows built, or None for the plain code."""
    if good is None:
        remove_recycled(pdir)
        return None
    from madmatrix.model_handling import OneProcessExporterMadMatrix
    nrows = OneProcessExporterMadMatrix.write_recycled_recipe(
        pdir, recipe, good, [tuple(pair) for pair in zeros])
    if nrows is None:
        remove_recycled(pdir)
    return nrows


def set_tag(pdir, tag):
    """The library-name tag of `pdir` (see the module docstring)."""
    path = pjoin(pdir, TAG_MK)
    if not tag:
        if os.path.exists(path):
            os.remove(path)
        return
    with open(path, 'w') as fsock:
        fsock.write('# --hel_recycling: this build is for another filter than the output\'s\n'
                    '# own (see HelRecyclingFilter.json), its library is named after it\n'
                    'HRTAG = %s\n' % tag)


def drop_objects(pdir):
    """The objects of `pdir`, of every backend: the build has no dependency on
    the recycled files, which have just changed."""
    for path in glob.glob(pjoin(pdir, 'build.*')):
        shutil.rmtree(path, ignore_errors=True)
    for path in glob.glob(pjoin(pdir, '*.o')) + glob.glob(pjoin(pdir, '.build.*')):
        os.remove(path)


def tagged_me_path(me_path, tag):
    """me_path ('lib/libmadmatrix_<model>_<P>_{device}_{fptype}.so') with tag."""
    return _TAG_RE.sub('', me_path).replace('_{device}', tag + '_{device}', 1)


def strip_tag(name):
    """A library name without its filter tag."""
    return re.sub(r'_hr[0-9a-f]{8}(?=_)', '', name)


# -- the two passes ----------------------------------------------------------

def measure_at_output(root, exporters, card):
    """The generation-time pass (madmatrix/output.py): measure every
    --hel_recycling P* directory of `exporters` at `card` and write its
    recycled build, or leave the plain code. Its filter is the output's own:
    the one whose library carries no tag."""
    tree = ProbeTree(root)
    tree.reset()
    key = cards_key([card])
    for exporter in exporters:
        pdir = exporter.path
        name = os.path.basename(pdir)
        recipe = exporter.write_hr_recipe()
        state = {'cards': key, 'recycled': False}
        try:
            good, zeros = tree.measure(pdir, [card])
            nrows = write_build(pdir, recipe, good, zeros)
            if nrows is not None:
                state.update(recycled=True, good_hels=good, zero_perhel=zeros)
                logger.info('hel_recycling: %s builds %d helicity rows of %d, '
                            'dropping %d (helicity, amplitude) pairs that vanish',
                            name, nrows, len(recipe['helicities']), len(zeros))
        except Exception as error:
            remove_recycled(pdir)
            logger.warning('hel_recycling warm-up error in %s (%s); it is '
                           'written without recycling.', name, error)
        state['output'] = signature(state)
        write_state(pdir, state)


def update(root, cards, pdirs, nb_core=None, subprocesses_json=None, log=None):
    """The launch-time pass: make the build of every --hel_recycling P*
    directory of `pdirs` (relative to `root`) right at every one of `cards`.

    A directory whose filter was measured at these same cards is left alone;
    otherwise its probe is run at them, and the build is rewritten only when
    the filter comes out different -- under the library name of that filter,
    whose objects are dropped so that the next make rebuilds it. When the
    filter cannot be measured, the plain code is built: it is right at any
    card. subprocesses.json (if given) is updated to the new library names.
    Returns {pdir: new me_path} of the directories whose library changed
    (with subprocesses_json), or {pdir: tag}. Reports to `log` (the caller's
    logger, by default this module's). cards=None: parameters no card can
    say, the plain code is built."""
    log = log or logger
    key = ['plain'] if cards is None else cards_key(cards)
    tree = ProbeTree(root, nb_core)
    tags = {}
    for pdir in pdirs:
        path = pjoin(root, pdir)
        if not os.path.isfile(pjoin(path, RECIPE)):
            continue
        state = read_state(path)
        if state is None:
            continue
        if state.get('cards') == key:
            continue
        name = os.path.basename(os.path.normpath(pdir))
        good, zeros = None, None
        if cards is None:
            pass
        elif not tree.available():
            log.warning('hel_recycling: %s cannot be measured at these parameters '
                        '(no probe driver in this output); it is built without '
                        'recycling.', name)
        else:
            try:
                good, zeros = tree.measure(path, cards)
            except Exception as error:
                log.warning('hel_recycling: %s could not be measured at these '
                            'parameters (%s); it is built without recycling.',
                            name, error)
        new = dict(state, cards=key)
        if good is not None:
            new.update(recycled=True, good_hels=good, zero_perhel=zeros)
            if signature(new) == signature(state):
                write_state(path, new)
                log.info('hel_recycling: %s has the same filter at these parameters '
                            '(%d rows, %d vanishing pairs): nothing to rebuild',
                            name, len(good), len(zeros))
                continue
        with open(pjoin(path, RECIPE)) as fsock:
            recipe = json.load(fsock)
        nrows = write_build(path, recipe, good, zeros)
        if nrows is None:
            new = dict(state, cards=key, recycled=False)
            new.pop('good_hels', None)
            new.pop('zero_perhel', None)
        if signature(new) == signature(state):
            write_state(path, new)
            continue
        tag = tag_of(new)
        set_tag(path, tag)
        drop_objects(path)
        write_state(path, new)
        if nrows is None:
            log.info('hel_recycling: %s is built without recycling at these parameters',
                        name)
        else:
            log.info('hel_recycling: %s re-measured at these parameters: %d helicity '
                        'rows of %d, dropping %d (helicity, amplitude) pairs (it had %s); '
                        'its library is now %s', name, nrows, len(recipe['helicities']),
                        len(zeros), '%d rows and %d pairs' % (len(state['good_hels']),
                                                              len(state['zero_perhel']))
                        if state.get('recycled') else 'no recycling',
                        'the one of that filter (%s)' % tag if tag else "the output's own")
        tags[pdir] = tag
    if not subprocesses_json or not tags:
        return tags
    with open(subprocesses_json) as fsock:
        entries = json.load(fsock)
    me_paths = {}
    for entry in entries:
        pdir = entry.get('path')
        if pdir in tags:
            entry['me_path'] = tagged_me_path(entry['me_path'], tags[pdir])
            me_paths[pdir] = entry['me_path']
    with open(subprocesses_json, 'w') as fsock:
        json.dump(entries, fsock)
    return me_paths
