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
"""Consistency of the helicity-recycled standalone output
(`output standalone --hel_recycling=True`) against the standard one.

With --hel_recycling the exporter writes matrix_orig.f (the plain per-helicity
MATRIX), template_matrix.f (the SMATRIX/MATRIX driver) and hel_warmup.f (a
probe), then runs the madevent DAG rewriter (madgraph/madevent/hel_recycle.py)
over them: the helicity loop is unrolled, wavefunctions that do not depend on a
given external helicity are computed once and shared, each amplitude is split
into a P1N current plus a contraction, and the helicity rows the warm-up found
to be dead are dropped. The warm-up also measures the good rows of every
crossing (the recycled table is their union, since a crossed call reuses the
baked rows) and the C-parity pairing (the partner's |M|^2 is copied from its
representative instead of being recomputed).

None of that may change a number: for every phase-space point and flavor the
printed |M|^2 must agree with the standard standalone to round-off. Each process
below is therefore generated twice, both outputs are compiled (`make check`) and
run (`./check`), and the printed values are compared entry by entry. For a
process whose crossings are folded into a single directory, ./check also prints
the crossed matrix elements, so those are covered by the same comparison.
"""

from __future__ import absolute_import

import logging
import os
import re
import shutil
import subprocess
import tempfile
import unittest

import madgraph.interface.master_interface as cmd_interface
import madgraph.various.misc as misc

logger = logging.getLogger('madgraph.acceptance')
pjoin = os.path.join


def _sanitize(process):
    return re.sub(r'[^A-Za-z0-9]+', '_', process).strip('_').lower()


def hel_recycling_test_factory(process, model='sm', tolerance=1e-9, options=''):
    def test(self):
        self.check_process(process, model=model, tolerance=tolerance,
                           options=options)
    test.__name__ = 'test_%s' % _sanitize(process)
    test.__doc__ = ('Check --hel_recycling and the standard standalone agree '
                    'on |M|^2 for %s.' % process)
    return test


class StandaloneHelRecyclingConsistency(unittest.TestCase):

    debugging = getattr(unittest, 'debug', False)

    @classmethod
    def setUpClass(cls):
        # everything here needs a working fortran compiler (make check).
        if not misc.which('gfortran') and not misc.which('f77'):
            raise unittest.SkipTest('no fortran compiler available')

    def setUp(self):
        self.cmd = cmd_interface.MasterCmd()
        self.cmd.no_notification()
        self.tmpdir = tempfile.mkdtemp(prefix='amc_helrecycling_')
        self.std_dir = pjoin(self.tmpdir, 'Standard')
        self.hr_dir = pjoin(self.tmpdir, 'Recycled')

    def tearDown(self):
        if not self.debugging and os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def do(self, line):
        self.cmd.exec_cmd(line)

    # ------------------------------------------------------------------
    # generation helpers
    # ------------------------------------------------------------------
    def _generate_pair(self, process, model='sm', options=''):
        """Write both outputs and return their (sorted) subprocess dir lists."""
        self.do('set automatic_html_opening False')
        self.do('set group_subprocesses False')
        self.do('import model %s' % model)
        self.do(('generate %s %s' % (process, options)).strip())
        self.do('output standalone %s -f' % self.std_dir)
        self.do('output standalone %s --hel_recycling=True -f' % self.hr_dir)

        std_subdirs = self._subprocess_dirs(self.std_dir)
        hr_subdirs = self._subprocess_dirs(self.hr_dir)
        self.assertEqual([os.path.basename(d) for d in std_subdirs],
                         [os.path.basename(d) for d in hr_subdirs],
                         'Different subprocess structure for %s' % process)
        return std_subdirs, hr_subdirs

    def _subprocess_dirs(self, outdir):
        root = pjoin(outdir, 'SubProcesses')
        dirs = [pjoin(root, name) for name in sorted(os.listdir(root))
                if name.startswith('P') and os.path.isdir(pjoin(root, name))]
        self.assertTrue(dirs, 'No subprocess directory found in %s' % root)
        return dirs

    def _run_standalone(self, subproc_dir):
        """Compile and run ./check, returning the printed |M|^2 values."""
        retcode = self._call(['make', 'check'], subproc_dir)
        self.assertEqual(retcode, 0,
                         'Failed to compile the standalone check in %s'
                         % subproc_dir)
        output = subprocess.Popen(['./check'], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  cwd=subproc_dir).communicate()[0].decode()
        values = [float(m.group('value')) for m in re.finditer(
            r'Matrix element\s*=\s*(?P<value>[\d\.eEdD\+-]+)',
            output.replace('D', 'E').replace('d', 'e'))]
        self.assertTrue(values, 'No matrix element printed by ./check in %s:\n%s'
                        % (subproc_dir, output))
        return values

    @staticmethod
    def _call(command, cwd):
        if logger.isEnabledFor(logging.INFO):
            return subprocess.call(command, cwd=cwd)
        with open(os.devnull, 'w') as devnull:
            return subprocess.call(command, stdout=devnull, stderr=devnull,
                                   cwd=cwd)

    # ------------------------------------------------------------------
    # the actual check
    # ------------------------------------------------------------------
    def check_process(self, process, model='sm', tolerance=1e-9, options=''):
        std_subdirs, hr_subdirs = self._generate_pair(process, model, options)

        for std_sub, hr_sub in zip(std_subdirs, hr_subdirs):
            # the rewriter really ran: it tags every emitted call with its
            # reuse count, which no other standalone template carries.
            with open(pjoin(hr_sub, 'matrix.f')) as fsock:
                recycled = fsock.read()
            self.assertTrue(re.search(r'!\s+count\s+\d', recycled),
                            'matrix.f in %s was not produced by the DAG rewriter'
                            % hr_sub)

            std_me = self._run_standalone(std_sub)
            hr_me = self._run_standalone(hr_sub)
            self.assertEqual(
                len(std_me), len(hr_me),
                'Different number of matrix elements for %s (%s): '
                'standard=%s recycled=%s'
                % (process, os.path.basename(std_sub), len(std_me), len(hr_me)))
            for i, (std_val, hr_val) in enumerate(zip(std_me, hr_me)):
                scale = max(abs(std_val), abs(hr_val), 1e-99)
                self.assertLessEqual(
                    abs(std_val - hr_val) / scale, tolerance,
                    'Incompatible |M|^2 for %s (%s, entry %s): standard=%s '
                    'recycled=%s' % (process, os.path.basename(std_sub), i,
                                     std_val, hr_val))


class TestStandaloneHelRecyclingConsistency(StandaloneHelRecyclingConsistency):

    # single topology, combined (gamma + Z = FFV6_2) routines
    test_helrec_ee_mumu = hel_recycling_test_factory('e+ e- > mu+ mu-')

    # cross-topology + non-trivial color + crossings folded into one directory
    test_helrec_uux_ddxg = hel_recycling_test_factory('u u~ > d d~ g')

    # 1 > 2 decay with a scalar external
    test_helrec_h_bbx = hel_recycling_test_factory('h > b b~')

    # identical final state particles (BROKEN_SYM must survive the rewrite)
    test_helrec_uux_uux = hel_recycling_test_factory('u u~ > u u~')

    # merged flavor: several coupling groups behind one matrix element
    test_helrec_pp_epem = hel_recycling_test_factory('p p > e+ e-')

    # massive (3-state) external vectors: no C-parity pairing is possible here
    test_helrec_uux_wpwm = hel_recycling_test_factory('u u~ > w+ w-')

    # polarization restriction: NCOMB is a non-contiguous subset of the
    # canonical helicity codes
    test_helrec_uux_wp0wm = hel_recycling_test_factory('u u~ > w+{0} w-')

    def test_c_parity_pairs_are_reused(self):
        """g g > t t~ has no 0-helicity state, so every row is paired with a
        distinct C-parity partner and the recycled file must copy the partner's
        |M|^2 rather than recompute it."""
        _, hr_subdirs = self._generate_pair('g g > t t~')
        with open(pjoin(hr_subdirs[0], 'matrix.f')) as fsock:
            recycled = fsock.read()
        reuse = re.findall(r'TS\((\d+)\)\s*=\s*TS\((\d+)\)', recycled)
        self.assertTrue(reuse,
                        'No C-parity reuse emitted for g g > t t~:\n%s'
                        % recycled)
        # the pairing is an involution on distinct rows
        for flip, rep in reuse:
            self.assertNotEqual(flip, rep)

    def test_zero_helicity_state_disables_the_reuse(self):
        """u u~ > w+ w- has 0-helicity rows that are their own C-parity
        partner, which makes the all-or-nothing reuse inapplicable."""
        _, hr_subdirs = self._generate_pair('u u~ > w+ w-')
        with open(pjoin(hr_subdirs[0], 'matrix.f')) as fsock:
            recycled = fsock.read()
        self.assertFalse(re.findall(r'TS\((\d+)\)\s*=\s*TS\((\d+)\)', recycled),
                         'C-parity reuse must not be applied to a process with '
                         'a self-paired helicity row')
