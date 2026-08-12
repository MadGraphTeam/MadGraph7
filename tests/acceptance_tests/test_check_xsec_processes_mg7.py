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
"""Systematic MG7 cross-section regression tests.

For every process listed in ``check_xsec_processes_reference.json`` this
module generates a ``test_<section>_<id>_mg7`` method on a single
``unittest.TestCase`` (so the whole suite is discoverable and launchable
through ``tests/test_manager.py``, exactly like the other mg7 acceptance
tests). Each test:

  1. builds an MG5 interface and ``generate``s the process (the ``_quark`` /
     ``_anti_quark`` merged-flavor particles are used directly from the sm
     model),
  2. ``output mg7``s it,
  3. edits ``Cards/run_card.toml`` -- it pins the fixed scale the references
     were generated with (mu = 91.188 GeV, e_cm = 13000 GeV,
     NNPDF23_lo_as_0130_qed), since the template now defaults to the dynamical
     HT/2 scale; it also sets the event count and, for the hadronic tt~ decays,
     neutralises the jet cuts (see CLAUDE.md),
  4. runs ``bin/generate_events -f`` and reads the cross-section from the
     madspace ``Events/*/info.json`` (``process.mean`` / ``process.error``),
  5. asserts the relative difference to the reference stays within a tolerance.

The source of truth is ``check_xsec_processes_reference.json`` (mirrors the
table in CLAUDE.md, produced with fixed scale and 1M events).

Two knobs are read from the environment so the CI can dial them without
touching the code:

  * ``MG7_XSEC_TOLERANCE`` -- max allowed relative difference (default 0.01, 1%)
  * ``MG7_XSEC_EVENTS``    -- events per run (default 100000; the reference
                             used 1M, reduced here to keep the CI affordable)

Run everything locally with e.g.::

    ./tests/test_manager.py test_.*_mg7 -pA -t0 -l INFO

or a single section::

    ./tests/test_manager.py test_ttx_decay_.*_mg7 -pA -t0

Each test self-skips (rather than failing) when the mg7 runtime stack
(madspace + LHAPDF + the NNPDF23 grid) is not available.
"""

from __future__ import absolute_import
from __future__ import division

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import unittest

import madgraph.interface.master_interface as MGCmd

pjoin = os.path.join
_HERE = os.path.dirname(os.path.abspath(__file__))
_REFERENCE = pjoin(_HERE, 'check_xsec_processes_reference.json')

# Environment-tunable knobs (see module docstring). Kept as module globals so
# the dynamically generated test methods pick up the CI-provided values.
_TOLERANCE = float(os.environ.get('MG7_XSEC_TOLERANCE', 0.01))
_EVENTS = int(os.environ.get('MG7_XSEC_EVENTS', 100000))

# Optional: when set (by the CI workflow), one JSON result record per process
# is written here so a later job can build a GitHub Actions job summary out of
# them. The per-process run directory (and its log) is torn down at the end of
# each test, so anything the summary needs -- including the tail of the log on
# failure -- must be captured into this record before tearDown() runs.
_RESULTS_DIR = os.environ.get('MG7_XSEC_RESULTS_DIR')


# madspace renders a live-updating terminal UI (cursor moves, line clears,
# colors) into the log; strip that so the tail is readable once scraped into
# a plain-text summary.
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b[()][A-Za-z0-9]|\x1b[78]')


def _tail(path, n=60):
    """Return the last *n* (ANSI-stripped) lines of *path*, best effort."""
    try:
        with open(path) as f:
            lines = f.readlines()
        return _ANSI_RE.sub('', ''.join(lines[-n:])).strip()
    except Exception:
        return ''


def _mg7_datadir_or_skip(test):
    """Return an LHAPDF data dir that contains the NNPDF23_lo_as_0130_qed set,
    or ``skipTest`` (on *test*) when the mg7 runtime stack (madspace + LHAPDF +
    the run_card.toml default PDF) is unavailable."""
    try:
        import madspace
        has_mg7 = hasattr(madspace, 'ChannelEventGenerator')
    except ImportError:
        has_mg7 = False
    if not has_mg7:
        test.skipTest('mg7 runtime stack (madspace) unavailable')

    candidates = []
    if os.environ.get('LHAPDF_DATA_PATH'):
        candidates.extend(os.environ['LHAPDF_DATA_PATH'].split(os.pathsep))
    try:
        out = subprocess.check_output(['lhapdf-config', '--datadir'],
                                      stderr=subprocess.DEVNULL).decode().strip()
        if out:
            candidates.append(out)
    except Exception:
        pass
    for d in candidates:
        if d and os.path.isdir(d) and glob.glob(pjoin(d, 'NNPDF23_lo_as_0130_qed*')):
            return d
    test.skipTest('NNPDF23_lo_as_0130_qed LHAPDF data not found '
                  '(set $LHAPDF_DATA_PATH)')


def _edit_run_card(toml_path, events, disable_jet_cuts):
    """Set the event count and (optionally) neutralise the jet cuts.

    The reference cross-sections were produced with FIXED scales
    (mu = 91.188 GeV). The run_card.toml template now defaults to the
    dynamical HT/2 scale instead, so these replacements are what pins the
    configuration back to the one the references were generated with. They
    are load-bearing: drop them and every reference value below goes stale."""
    t = open(toml_path).read()
    t = t.replace('fixed_ren_scale = false', 'fixed_ren_scale = true')
    t = t.replace('fixed_fact_scale = false', 'fixed_fact_scale = true')
    t = re.sub(r'events = \d+', 'events = %d' % events, t)
    if disable_jet_cuts:
        # jet cuts must be disabled for the hadronic tt~ decay processes to
        # match the reference (see CLAUDE.md).
        t = re.sub(r'jet-pt\.min\s*=.*', 'jet-pt.min = 0.0', t)
        t = re.sub(r'jet-eta_abs\.max\s*=.*', 'jet-eta_abs.max = 100.0', t)
        t = re.sub(r'jet-delta_r\.min\s*=.*', 'jet-delta_r.min = 0.0', t)
        t = re.sub(r'jet-lepton-delta_r\.min\s*=.*', 'jet-lepton-delta_r.min = 0.0', t)
    open(toml_path, 'w').write(t)


with open(_REFERENCE) as _f:
    _REF = json.load(_f)


class CheckXsecProcessesMG7Test(unittest.TestCase):
    """Regression test: MG7 (madspace) reproduces the recorded reference
    cross-sections within ``MG7_XSEC_TOLERANCE``. One ``test_*`` method per
    process is attached below from ``check_xsec_processes_reference.json``."""

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='check_xsec_mg7_')

    def tearDown(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def _record_result(self, entry, section, status, got=None, err=None,
                        message=None):
        """Persist a machine-readable record of this process' outcome (used
        by the CI workflow to build a job summary). No-op unless
        ``MG7_XSEC_RESULTS_DIR`` is set."""
        if not _RESULTS_DIR:
            return
        try:
            os.makedirs(_RESULTS_DIR, exist_ok=True)
            record = {
                'section': section,
                'id': entry['id'],
                'process': entry['process'],
                'status': status,
                'cross': got,
                'error': err,
                'ref_cross': entry['cross'],
                'ref_error': entry.get('error'),
                'message': message,
            }
            out = pjoin(_RESULTS_DIR, '%s_%s.json' % (section, entry['id']))
            with open(out, 'w') as f:
                json.dump(record, f)
        except Exception:
            # Recording the result must never mask the actual test outcome.
            pass

    def _check_process(self, entry, defines, section):
        """Generate + integrate a single process and assert its cross-section
        matches the reference within tolerance."""
        try:
            self._run_and_check(entry, defines, section)
        except unittest.SkipTest:
            raise
        except AssertionError:
            # Already recorded (with a precise message) by _run_and_check.
            raise
        except Exception:
            self._record_result(entry, section, 'fail',
                                 message=traceback.format_exc())
            raise

    def _run_and_check(self, entry, defines, section):
        datadir = _mg7_datadir_or_skip(self)

        run_dir = pjoin(self.path, entry['id'])
        mg = MGCmd.MasterCmd()
        mg.no_notification()
        mg.exec_cmd('set automatic_html_opening False --no_save')
        for d in defines:
            mg.exec_cmd(d)
        mg.exec_cmd('generate %s' % entry['process'])
        mg.exec_cmd('output mg7 %s' % run_dir)

        toml = pjoin(run_dir, 'Cards', 'run_card.toml')
        _edit_run_card(toml, _EVENTS, entry.get('disable_jet_cuts', False))

        env = dict(os.environ)
        env['LHAPDF_DATA_PATH'] = datadir
        log = pjoin(run_dir, 'mg7_gen.log')
        with open(log, 'w') as logfh:
            ret = subprocess.call(
                [sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f'],
                cwd=run_dir, env=env, stdout=logfh, stderr=subprocess.STDOUT)
        if ret != 0:
            message = ('mg7 generate_events failed (exit %d)\n\n%s'
                       % (ret, _tail(log)))
            self._record_result(entry, section, 'fail', message=message)
            self.fail('mg7 generate_events failed (see %s)' % log)

        infos = sorted(glob.glob(pjoin(run_dir, 'Events', '*', 'info.json')))
        if not infos:
            message = 'no mg7 info.json produced\n\n%s' % _tail(log)
            self._record_result(entry, section, 'fail', message=message)
            self.fail('no mg7 info.json under %s (see %s)' % (run_dir, log))
        with open(infos[-1]) as infofh:
            info = json.load(infofh)['process']
        got = float(info['mean'])
        err = float(info.get('error') or 0.0)

        ref_x = entry['cross']
        reldiff = abs(got - ref_x) / ref_x if ref_x else float('inf')
        passed = reldiff <= _TOLERANCE
        message = None
        if not passed:
            # A cross-section was successfully obtained here, just outside
            # tolerance -- no need for the (noisy) log tail, the deviation
            # itself is the useful diagnostic.
            message = (
                '%s (%s): mg7 xsec %.6g +- %.3g pb differs from reference '
                '%.6g pb by %.3f%% (> %.3f%% tolerance)'
                % (entry['id'], entry['process'], got, err, ref_x,
                   100 * reldiff, 100 * _TOLERANCE))
        self._record_result(entry, section, 'pass' if passed else 'fail',
                             got=got, err=err, message=message)
        self.assertLessEqual(reldiff, _TOLERANCE, message)


def _make_test(entry, defines, section):
    """Build a bound-method-shaped closure for a single reference entry."""
    def test(self):
        self._check_process(entry, defines, section)
    test.__doc__ = '%s: %s (ref %.6g pb)' % (
        entry['id'], entry['process'], entry['cross'])
    return test


# Attach one test_<section>_<id>_mg7 method per process. The section is part of
# the method name so the CI (and a developer) can select a single section with
# a regexp, e.g. test_manager.py test_ttx_decay_.*_mg7 -pA
for _section, _entries in _REF['sections'].items():
    for _entry in _entries:
        _name = 'test_%s_%s_mg7' % (_section, _entry['id'])
        setattr(CheckXsecProcessesMG7Test, _name,
                _make_test(_entry, _REF['defines'], _section))

if __name__ == '__main__':
    unittest.main()
