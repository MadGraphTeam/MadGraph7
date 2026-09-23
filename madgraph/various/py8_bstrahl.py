################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
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
"""Locate, build and select the Pythia8 main that handles beamstrahlung.

A dressed-lepton density convolutes beamstrahlung with ISR, so the momentum
fraction in an event file is a product, x = xbs * xisr.  Showering such a
file with an ordinary main leaves the beams at their nominal energy and
hands the whole of 1-x to the ISR shower, which recoils the hard system in
a way beamstrahlung does not.  vendor/py8main/mg5amc_bstrahl.cc consumes
the per-event <beamstrahlung> tag instead; this module builds it and works
out when it is needed.
"""

import logging
import os
import shutil
import subprocess

# Same two contexts as the rest of the run interface: a process directory
# ships this module into bin/internal, where the package is 'internal'.
try:
    import madgraph.various.misc as misc
except ImportError:
    import internal.misc as misc

logger = logging.getLogger('madgraph.py8_bstrahl')

pjoin = os.path.join

MAIN_NAME = 'mg5amc_bstrahl'
MAKEFILE_NAME = 'Makefile.mgbstrahl'


def source_dir(mg5dir):
    """Where the main and its makefile fragment are shipped."""
    return pjoin(mg5dir, 'vendor', 'py8main')


def examples_dir(pythia8_path):
    """The examples directory of a Pythia8 installation, or None.

    Both layouts that MadGraph already probes for main164 are accepted.
    """
    for candidate in (pjoin(pythia8_path, 'share', 'Pythia8', 'examples'),
                      pjoin(pythia8_path, 'examples')):
        if os.path.isdir(candidate):
            return candidate
    return None


def fallback_dir(mg5dir):
    """Where to build when the Pythia8 installation is not writable."""
    return pjoin(mg5dir, 'HEPTools', 'py8main')


def find_binary(mg5dir, pythia8_path):
    """Return the built main, or None if it has not been built yet."""
    candidates = []
    examples = examples_dir(pythia8_path)
    if examples:
        candidates.append(pjoin(examples, MAIN_NAME))
    if mg5dir:
        candidates.append(pjoin(fallback_dir(mg5dir), MAIN_NAME))
    for candidate in candidates:
        if os.path.exists(candidate) and misc.is_executable(candidate):
            return candidate
    return None


def get_main(pythia8_path, mg5dir=None):
    """The beamstrahlung main to shower with, or None.

    In a standalone process directory mg5dir is None and only an already
    built main is accepted: the source lives in the MadGraph tree, and the
    binary is normally produced once when Pythia8 is installed.
    """
    existing = find_binary(mg5dir, pythia8_path)
    if existing or mg5dir is None:
        return existing
    return build(mg5dir, pythia8_path)


def build(mg5dir, pythia8_path, force=False):
    """Build the beamstrahlung main against this Pythia8; return its path.

    Preferred location is the Pythia8 examples directory, so that Pythia's
    own Makefile.inc supplies the compiler flags and the optional HepMC
    settings. A read-only installation is common enough that we fall back
    to building under HEPTools instead, pointing the fragment back at the
    installation.

    Returns None on failure, after logging why: the caller decides whether
    that is fatal.
    """
    if not force:
        existing = find_binary(mg5dir, pythia8_path)
        if existing:
            return existing

    src = source_dir(mg5dir)
    if not os.path.isdir(src):
        logger.warning('Cannot build %s: %s is missing.', MAIN_NAME, src)
        return None

    targets = []
    examples = examples_dir(pythia8_path)
    if examples and os.access(examples, os.W_OK):
        targets.append((examples, []))
    targets.append((fallback_dir(mg5dir),
                    ['PYTHIA8=%s' % os.path.abspath(pythia8_path)]))

    for target, extra in targets:
        binary = pjoin(target, MAIN_NAME)
        try:
            if not os.path.isdir(target):
                os.makedirs(target)
            for name in (MAIN_NAME + '.cc', MAKEFILE_NAME):
                shutil.copy(pjoin(src, name), target)
            # A binary left by an earlier Pythia8 must not be mistaken for
            # the product of this build: HEPTools/py8main is never cleaned,
            # and a main linked against another libpythia8 is worse than no
            # main at all, since the caller treats a returned path as usable.
            if os.path.exists(binary):
                os.remove(binary)
            log = pjoin(target, MAIN_NAME + '.log')
            with open(log, 'w') as fsock:
                status = misc.call(['make', '-f', MAKEFILE_NAME] + extra,
                                   cwd=target, stdout=fsock,
                                   stderr=subprocess.STDOUT)
        except (IOError, OSError) as error:
            logger.debug('Building %s in %s failed: %s',
                         MAIN_NAME, target, error)
            continue
        if status != 0:
            logger.debug('make returned %s while building %s in %s; see %s',
                         status, MAIN_NAME, target, log)
            continue
        if os.path.exists(binary) and misc.is_executable(binary):
            logger.info('Built the beamstrahlung Pythia8 main in %s', target)
            return binary
        logger.debug('Building %s in %s produced no binary; see %s',
                     MAIN_NAME, target, log)

    logger.warning(
        'Could not build %s against the Pythia8 installation in %s. '
        'The build output was kept next to each attempted location, as %s.log.',
        MAIN_NAME, pythia8_path, MAIN_NAME)
    return None


def density_has_bstrahl(pdlabel, source_dir_path):
    """True if this lepton density convolutes beamstrahlung into its PDF.

    Asked of the density's own bstrahl.f, which is generated alongside the
    grid and is the same authority the Fortran uses at run time, so the two
    cannot drift apart.  Used at card-checking time, before any event file
    exists; once there are events, event_file_needs_bstrahl() is better
    because it describes the file actually being showered.
    """
    if not pdlabel:
        return False
    path = pjoin(source_dir_path, 'PDF', 'lep_densities', str(pdlabel),
                 'bstrahl.f')
    if not os.path.exists(path):
        return False
    try:
        with open(path) as fsock:
            for line in fsock:
                if 'eepdf_has_bstrahl' in line and '.true.' in line.lower():
                    return True
    except IOError:
        pass
    return False


def event_file_needs_bstrahl(path):
    """True if this event file carries per-event beamstrahlung fractions.

    The file is the authority here rather than the run card: it is what the
    shower will actually read, and it stays correct for files that have
    been through reweighting or MadSpin, or that were produced by another
    version of the code.  Only the first event is inspected, since the tag
    is written for every event of a run or for none of them.
    """
    if not path or not os.path.exists(path):
        return False
    # imported here to keep this module importable from the installer,
    # which has no need for the event parser
    try:
        import madgraph.various.lhe_parser as lhe_parser
    except ImportError:
        import internal.lhe_parser as lhe_parser
    try:
        event_file = lhe_parser.EventFile(path)
        try:
            for event in event_file:
                return event.parse_beamstrahlung() is not None
        finally:
            event_file.close()
    except Exception as error:
        logger.debug('Could not inspect %s for beamstrahlung: %s',
                     path, error)
    return False
