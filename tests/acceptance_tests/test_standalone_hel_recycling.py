################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Consistency of the helicity-recycled standalone output
(`output standalone_fortran --hel_recycling=True`) against the standard one.

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

./check works in the partonic CM frame, which is also where the warm-up used to
measure its zeros; the recycled output is called in any frame, so the frame
checks below go through the f2py module with CM AND boosted points.
"""

from __future__ import absolute_import

import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import madgraph.interface.master_interface as cmd_interface
import madgraph.various.misc as misc

logger = logging.getLogger('madgraph.acceptance')
pjoin = os.path.join

# Evaluates one combined f2py module (--prefix=int) in its own interpreter --
# every output builds a module called all_matrix2py, and two of them in one
# process would share their common blocks. For the flavor index of spec['pdgs']
# it reports the helicity table, then at each point the summed SMATRIX (a
# recycled build runs its baked rows there; SMATRIXHEL(-1) is not summed by the
# recycled driver) and SMATRIXHEL for every helicity code.
_FRAME_PROBE = r'''
import json, os, sys
import numpy as np
os.chdir(%(subproc)r)
sys.path.insert(0, %(subproc)r)
import all_matrix2py as m

spec = json.load(open(%(spec)r))
m.initialise(os.path.abspath('../Cards/param_card.dat'))
prefix = m.get_prefix()[0].decode().strip().lower()

def entry(name):
    return getattr(m, 'py_' + prefix + name)

nflav, _, ncross = entry('get_flavor_layout')()
idx = [f for f in range(1, nflav * ncross + 1)
       if [int(x) for x in entry('get_pdg_for_flavor')(f)] == spec['pdgs']][0]
nhel = np.array(entry('get_nhel_idx')(idx)[1])
out = {'rows': [[int(h) for h in nhel[:, c]] for c in range(nhel.shape[1])],
       'summed': [], 'codes': []}
for p in spec['points']:
    mom = np.array(p).T
    out['summed'].append(float(entry('smatrix_idx')(mom, idx)))
    out['codes'].append([float(entry('smatrixhel_idx')(mom, c, idx))
                         for c in range(1, nhel.shape[1] + 1)])
print('PROBE ' + json.dumps(out))
'''


def _cm_point(sqrts, m3, m4, cos_theta, phi):
    """A 2 -> 2 point in the partonic CM frame, massless initial legs."""
    s = sqrts ** 2
    pf = math.sqrt((s - (m3 + m4) ** 2) * (s - (m3 - m4) ** 2)) / (2 * sqrts)
    sin_theta = math.sqrt(1 - cos_theta ** 2)
    v = [pf * sin_theta * math.cos(phi), pf * sin_theta * math.sin(phi),
         pf * cos_theta]
    return [[sqrts / 2, 0., 0., sqrts / 2], [sqrts / 2, 0., 0., -sqrts / 2],
            [math.sqrt(pf ** 2 + m3 ** 2)] + v,
            [math.sqrt(pf ** 2 + m4 ** 2)] + [-x for x in v]]


def _boost(point, beta):
    """`point` boosted by the velocity `beta`."""
    b2 = sum(b * b for b in beta)
    gamma = 1 / math.sqrt(1 - b2)
    boosted = []
    for e, px, py, pz in point:
        bp = beta[0] * px + beta[1] * py + beta[2] * pz
        k = (gamma - 1) / b2 * bp + gamma * e
        boosted.append([gamma * (e + bp), px + k * beta[0], py + k * beta[1],
                        pz + k * beta[2]])
    return boosted


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
        self.do('output standalone_fortran %s -f' % self.std_dir)
        self.do('output standalone_fortran %s --hel_recycling=True -f' % self.hr_dir)

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

    def test_full_entry_point_api_is_emitted(self):
        """Only SMATRIX/SMATRIXHEL/MATRIX take the recycled path; every other
        entry point is appended from the standard template, so the recycled
        output must expose the same API (the density stack in particular, which
        evaluates arbitrary helicity rows the recycled table cannot serve)."""
        _, hr_subdirs = self._generate_pair('p p > e+ e-')
        with open(pjoin(hr_subdirs[0], 'matrix.f')) as fsock:
            recycled = fsock.read().upper()
        for routine in ('GET_AMP', 'GET_JAMP', 'GET_MATRIX', 'GET_INTER',
                        'GET_DENSITY', 'GET_DENSITY_IDX', 'GET_ALL_INTER',
                        'GET_ALL_INTER_IDX', 'GET_VALUE', 'GET_VALUE_IDX',
                        'GET_NHEL', 'GET_NHEL_IDX', 'FILL_NHEL',
                        'DECODE_HEL', 'ENCODE_HEL'):
            self.assertTrue(
                re.search(r'SUBROUTINE\s+%s\s*\(' % routine, recycled),
                '%s missing from the recycled output' % routine)
        # ... and no leftover stub refusing to compute the density
        self.assertNotIn('NOT AVAILABLE WITH --HEL_RECYCLING', recycled)

    def test_density_does_not_use_the_c_parity_reuse(self):
        """The C-parity de-duplication is only valid for the helicity-summed
        |M|^2: it asserts |M(h)|^2 == |M(-h)|^2, which says nothing about the
        interference terms JAMP_i JAMP_j* the density matrix is built from.

        The reuse therefore has to stay confined to TS, which only SMATRIX and
        SMATRIXHEL read; the density stack must keep going through the plain
        per-helicity GET_AMP/GET_JAMP/GET_INTER.
        """
        _, hr_subdirs = self._generate_pair('g g > t t~')
        with open(pjoin(hr_subdirs[0], 'matrix.f')) as fsock:
            source = fsock.read()
        # the reuse is emitted for this process (no 0-helicity state)
        self.assertTrue(re.findall(r'TS\((\d+)\)\s*=\s*TS\((\d+)\)', source),
                        'expected C-parity reuse for g g > t t~')
        # every routine of the density stack must be TS-free
        routines = re.split(r'\n(?=\s*(?:SUBROUTINE|DOUBLE PRECISION FUNCTION'
                            r'|INTEGER FUNCTION|REAL\*8 FUNCTION))', source)
        for body in routines:
            head = body.strip().split('\n')[0].upper()
            if re.search(r'\b(GET_DENSITY|GET_ALL_INTER|GET_INTER|GET_JAMP'
                         r'|GET_AMP|GET_MATRIX)\w*\s*\(', head):
                self.assertNotIn('TS(', body.upper(),
                                 'the C-parity reuse must not reach %s' % head)

    def test_zero_helicity_state_disables_the_reuse(self):
        """u u~ > w+ w- has 0-helicity rows that are their own C-parity
        partner, which makes the all-or-nothing reuse inapplicable."""
        _, hr_subdirs = self._generate_pair('u u~ > w+ w-')
        with open(pjoin(hr_subdirs[0], 'matrix.f')) as fsock:
            recycled = fsock.read()
        self.assertFalse(re.findall(r'TS\((\d+)\)\s*=\s*TS\((\d+)\)', recycled),
                         'C-parity reuse must not be applied to a process with '
                         'a self-paired helicity row')

    def test_crossed_smatrixhel(self):
        """SMATRIXHEL at a crossed FLAV_IDX under --hel_recycling.

        Every crossing evaluates the baked base rows (each helicity times IC,
        as APPLY_CROSSING_ROWS feeds the standard SMATRIX), so the crossed
        process's own helicity code selects the base row CROSS_HELCODE names:
        the recycled output used to refuse a crossed index. `u b1 > c1 d1`
        (b1 = g u~, c1 = u z, d1 = z g) folds u u~ > z g onto u g > u z, a
        crossing that moves the massive z (whose helicity zeros depend on the
        frame, so its rows come from the warm-up's scan of the crossing):
        code by code, the recycled crossed |M|^2 must equal the standard
        standalone's. HEL=-1 names no helicity and asks for the sum, as in
        the standard SMATRIXHEL: the recycled one used to answer 0 for it."""
        import math
        from tests.acceptance_tests.test_standalone_cross_symmetry import \
            TestStandaloneCrossSymmetry
        self.do('set automatic_html_opening False')
        self.do('set group_subprocesses False')
        self.do('import model sm')
        for line in ('define b1 = g u~', 'define c1 = u z', 'define d1 = z g'):
            self.do(line)
        self.do('generate u b1 > c1 d1 QED=1 QCD=1')
        self.do('output standalone_fortran %s -f' % self.std_dir)
        self.do('output standalone_fortran %s --hel_recycling=True -f'
                % self.hr_dir)

        # u u~ > z g at sqrt(s) = 1 TeV, theta = 0.7: the z is leg 3
        mz, energy, theta = 91.188, 500.0, 0.7
        pz = (4 * energy ** 2 - mz ** 2) / (4 * energy)
        ez = math.sqrt(pz ** 2 + mz ** 2)
        sin, cos = math.sin(theta), math.cos(theta)
        momenta = [(energy, 0, 0, energy), (energy, 0, 0, -energy),
                   (ez, pz * sin, 0, pz * cos), (pz, -pz * sin, 0, -pz * cos)]
        stdin = '\n'.join(['2 24'] + [' '.join('%.17e' % x for x in mom)
                                       for mom in momenta]) + '\n'
        per_code = {}
        for name, root in (('standard', self.std_dir), ('recycled', self.hr_dir)):
            found = [d for d in self._subprocess_dirs(root)
                     if d.endswith('_ug_uz')]
            self.assertEqual(len(found), 1, 'u g > u z in %s: %s' % (root, found))
            pdir = found[0]
            driver = TestStandaloneCrossSymmetry._HELCODE_DRIVER
            tail = '      ENDDO\n      END\n'
            self.assertTrue(driver.endswith(tail))
            driver = driver[:-len('      END\n')] + (
                "      CALL SMATRIXHEL(P, -1, FLAV_IDX, ANS)\n"
                "      WRITE(*,'(A,ES24.16)') ' HELSUM=', ANS\n"
                "      CALL SMATRIX(P, FLAV_IDX, ANS)\n"
                "      WRITE(*,'(A,ES24.16)') ' SMATRIX=', ANS\n"
                "      END\n")
            with open(pjoin(pdir, 'check_sa.f'), 'w') as fsock:
                fsock.write(driver)
            self.assertEqual(self._call(['make', 'check'], pdir), 0,
                             'failed to build the %s driver' % name)
            out = subprocess.run(['./check'], cwd=pdir, input=stdin,
                                 capture_output=True, text=True).stdout
            per_code[name] = [float(v) for v in
                              re.findall(r'HEL=\s*\d+\s+(\S+)', out)]
            self.assertEqual(len(per_code[name]), 24,
                             '%s: no per-code output:\n%s' % (name, out))
            helsum = [float(v) for v in re.findall(r'HELSUM=\s*(\S+)', out)]
            smatrix = [float(v) for v in re.findall(r'SMATRIX=\s*(\S+)', out)]
            self.assertEqual(len(helsum), 1, out)
            self.assertEqual(len(smatrix), 1, out)
            self.assertGreater(smatrix[0], 0)
            self.assertAlmostEqual(helsum[0], smatrix[0],
                                   delta=1e-9 * smatrix[0],
                                   msg='%s: SMATRIXHEL(-1) %r, SMATRIX %r'
                                   % (name, helsum[0], smatrix[0]))
            per_code[name + '_sum'] = smatrix[0]
        self.assertEqual(sum(1 for v in per_code['standard'] if v), 12)
        self.assertAlmostEqual(per_code['recycled_sum'],
                               per_code['standard_sum'],
                               delta=1e-9 * per_code['standard_sum'])
        for code, (a, b) in enumerate(zip(per_code['recycled'],
                                          per_code['standard']), 1):
            self.assertAlmostEqual(a, b, delta=1e-9 * max(abs(b), 1e-300),
                                   msg='code %d: recycled %r, standard %r'
                                   % (code, a, b))

    # ------------------------------------------------------------------
    # frame independence: the warm-up must not bake a zero of the CM frame
    # ------------------------------------------------------------------
    # CM points, and the velocities they are boosted by. None of the velocities
    # is one the warm-up itself boosts by (hel_warmup_v4.inc BOOSTV).
    FRAME_ANGLES = ((0.3, 0.7), (-0.55, 2.1), (0.8, 4.0))
    FRAME_BOOSTS = ((0.15, -0.1, 0.6), (-0.4, -0.15, -0.45))

    def _f2py_module(self, outdir):
        """Build the combined f2py module of `outdir`; skip without f2py."""
        subproc = pjoin(outdir, 'SubProcesses')
        if self._call(['make', 'all_matrix2py.so'], subproc) != 0 or not any(
                f.startswith('all_matrix2py') and f.endswith('.so')
                for f in os.listdir(subproc)):
            raise unittest.SkipTest('could not build the f2py module in %s'
                                    % subproc)
        return subproc

    def _frame_probe(self, subproc, pdgs, points):
        spec = pjoin(self.tmpdir, 'frame_probe_%d.json'
                     % len(os.listdir(self.tmpdir)))
        with open(spec, 'w') as fsock:
            json.dump({'pdgs': pdgs, 'points': points}, fsock)
        script = spec[:-len('.json')] + '.py'
        with open(script, 'w') as fsock:
            fsock.write(_FRAME_PROBE % {'subproc': subproc, 'spec': spec})
        output = subprocess.run([sys.executable, script], cwd=subproc,
                                capture_output=True, text=True)
        lines = [l for l in output.stdout.splitlines() if l.startswith('PROBE ')]
        self.assertTrue(lines, 'no result from the f2py probe:\n%s%s'
                        % (output.stdout, output.stderr))
        return json.loads(lines[-1][len('PROBE '):])

    @staticmethod
    def _masses(outdir):
        """PDG -> mass, from the output's param_card."""
        masses, block = {}, None
        with open(pjoin(outdir, 'Cards', 'param_card.dat')) as fsock:
            for line in fsock:
                words = line.split('#')[0].split()
                if not words:
                    continue
                if words[0].lower() == 'block':
                    block = words[1].lower()
                elif block == 'mass' and len(words) == 2:
                    masses[int(words[0])] = float(words[1])
        return masses

    def check_frame_independence(self, process, pdgs, settings=()):
        """The summed |M|^2 of the recycled output, and every helicity code of
        it, must equal the standard output's at CM points AND at the same points
        boosted. The reference is the standard output's per-code SMATRIXHEL,
        which opens every helicity row (no training, no C-parity reuse).

        Returns (standard, recycled) probe results: 'points' lists the CM points
        first, then the boosted ones."""
        self.do('set automatic_html_opening False')
        for line in settings:
            self.do(line)
        self.do('import model sm')
        self.do('generate %s --use_crossing=False' % process)
        self.do('output standalone_fortran %s --prefix=int -f' % self.std_dir)
        self.do('output standalone_fortran %s --prefix=int --hel_recycling=True'
                ' -f' % self.hr_dir)
        std_sub = self._f2py_module(self.std_dir)
        hr_sub = self._f2py_module(self.hr_dir)

        masses = self._masses(self.std_dir)
        cm = [_cm_point(500., masses.get(abs(pdgs[2]), 0.),
                        masses.get(abs(pdgs[3]), 0.), c, phi)
              for c, phi in self.FRAME_ANGLES]
        points = cm + [_boost(p, beta) for beta in self.FRAME_BOOSTS
                       for p in cm]
        std = self._frame_probe(std_sub, pdgs, points)
        hr = self._frame_probe(hr_sub, pdgs, points)
        self.assertEqual(std['rows'], hr['rows'])
        # SMATRIXHEL multiplies by the number of initial helicity states
        helavg = len(set(r[0] for r in std['rows'])) * \
            len(set(r[1] for r in std['rows']))
        for i, (std_codes, hr_codes, hr_sum) in enumerate(
                zip(std['codes'], hr['codes'], hr['summed'])):
            frame = 'CM' if i < len(cm) else 'boosted'
            exact = sum(std_codes) / helavg
            self.assertGreater(exact, 0.)
            self.assertLessEqual(
                abs(hr_sum - exact), 1e-9 * exact,
                '%s, %s point %d: recycled summed %r, standard %r (ratio %r)'
                % (process, frame, i, hr_sum, exact, hr_sum / exact))
            scale = max(abs(v) for v in std_codes)
            for code, (a, b) in enumerate(zip(hr_codes, std_codes), 1):
                self.assertLessEqual(
                    abs(a - b), 1e-9 * scale,
                    '%s, %s point %d, code %d: recycled %r, standard %r'
                    % (process, frame, i, code, a, b))
        std['points'] = hr['points'] = points
        return std, hr

    def _live_rows(self, codes):
        """How many helicity codes carry |M|^2 above round-off at a point."""
        scale = max(abs(v) for v in codes)
        return sum(1 for v in codes if abs(v) > 1e-12 * scale)

    def _recycled_file(self, name):
        found = self._subprocess_dirs(self.hr_dir)
        self.assertEqual(len(found), 1, found)
        with open(pjoin(found[0], name)) as fsock:
            return fsock.read()

    def _recycled_reuse(self):
        """The C-parity reuse lines of the recycled matrix.f."""
        return re.findall(r'TS\((\d+)\)\s*=\s*TS\((\d+)\)',
                          self._recycled_file('matrix.f'))

    def _warmup_hmv(self):
        """The legs the warm-up closes the good rows over (DATA HMV)."""
        data = re.search(r'DATA HMV\s*/(.*?)/',
                         self._recycled_file('hel_warmup.f').replace('\n     &', ''),
                         re.S).group(1)
        return [v.strip().upper() == '.TRUE.' for v in data.split(',')]

    def test_frame_independence_massive_vector(self):
        """g u > z u (unfolded): rows 9, 11, 14 and 16 vanish in the partonic
        CM frame only -- the z's helicity states mix under a boost -- and
        boosted they carry up to 3% of the total. The warm-up used to scan CM
        points only, so the recycled output dropped those rows, and it also
        dropped the s-channel diagram from half of the rows it kept: that
        diagram vanishes in the CM frame by angular momentum along the beam
        axis, but a single diagram is not Lorentz invariant. Boosted by
        (0.15, -0.1, 0.6) the summed |M|^2 was off by up to 7%, either way."""
        std, _ = self.check_frame_independence('g u > z u', [21, 2, 23, 2])
        # the baked rows are closed over the z's helicity
        self.assertEqual(self._warmup_hmv(), [False, False, True, False])
        # the premise: 8 rows in the CM frame, 12 once boosted
        ncm = len(self.FRAME_ANGLES)
        self.assertEqual([self._live_rows(c) for c in std['codes'][:ncm]],
                         [8] * ncm)
        self.assertEqual([self._live_rows(c) for c in std['codes'][ncm:]],
                         [12] * (len(std['codes']) - ncm))

    def test_frame_independence_polarised_vector(self):
        """g u > w+{0} d: a polarised leg never crosses, so the directory is
        scanned by its own (uncrossed) warm-up only. The longitudinal
        polarization vector is defined in the frame of the call, and the
        CM-only warm-up kept a single row of the eight: boosted, the recycled
        |M|^2 was off by up to a factor 2000. Every row of the polarised
        table must match the standard output's in every frame."""
        std, _ = self.check_frame_independence('g u > w+{0} d',
                                               [21, 2, 24, 1])
        # the polarised table has the W in its longitudinal state only, so
        # there is no other W state to close over: the boosted scan alone
        # must keep the rows
        self.assertEqual(set(r[2] for r in std['rows']), {0})
        self.assertEqual(self._warmup_hmv(), [False] * 4)

    def test_frame_independence_massless(self):
        """g u > a u: no massive leg, so no row depends on the frame -- but the
        s-channel diagram still vanishes exactly in the CM frame for half of
        the rows (both incoming legs on the z axis), and a single diagram is
        not Lorentz invariant: a boost adds a gauge term to the polarization
        vectors HELAS builds for the gluon and the photon. The warm-up dropped
        it as a per-helicity zero, and the boosted |M|^2 came out up to 3%
        off. The closure has nothing to act on here: only the boosted scan
        keeps that diagram."""
        self.check_frame_independence('g u > a u', [21, 2, 22, 2])
        self.assertEqual(self._warmup_hmv(), [False] * 4)

    def test_frame_independence_massive_fermions(self):
        """g g > t t~ (unfolded): four rows vanish in the CM frame only (the
        tops are massive), and the s-channel diagram vanishes there for
        opposite gluon helicities. Its C-parity pairing, on the other hand,
        holds in every frame (QCD couplings are vector-like: a helicity flip is
        a complex conjugation), so the boosted validation keeps the reuse --
        and this checks it at boosted points too."""
        std, _ = self.check_frame_independence('g g > t t~', [21, 21, 6, -6])
        self.assertEqual(self._warmup_hmv(), [False, False, True, True])
        ncm = len(self.FRAME_ANGLES)
        self.assertEqual([self._live_rows(c) for c in std['codes'][:ncm]],
                         [12] * ncm)
        self.assertTrue(self._recycled_reuse(),
                        'the C-parity reuse of g g > t t~ holds in every frame '
                        'and must be kept')

    def test_frame_independence_cparity_with_width(self):
        """g g > t t~ keeping the top width in its t-channel propagators: the
        C-parity pairing |M(h)|^2 = |M(-h)|^2 then holds at 2 -> 2 CM points
        only (coplanar momenta: a reflection through the scattering plane), not
        once boosted (2.7e-3 apart at the first point). The warm-up compares
        the pairs at boosted points too and must refuse the reuse."""
        std, _ = self.check_frame_independence(
            'g g > t t~', [21, 21, 6, -6],
            settings=('set zerowidth_tchannel False --no_save',
                      'set zerowidth_external False --no_save'))
        # the premise: the pairing holds at the CM points, not at boosted ones
        flip = [std['rows'].index([-h for h in row]) for row in std['rows']]
        ncm = len(self.FRAME_ANGLES)
        mismatch = [max(abs(codes[c] - codes[flip[c]]) for c in range(len(flip)))
                    / max(codes) for codes in std['codes']]
        self.assertLess(max(mismatch[:ncm]), 1e-12, mismatch)
        self.assertGreater(max(mismatch[ncm:]), 1e-6, mismatch)
        self.assertFalse(self._recycled_reuse(),
                         'C-parity reuse kept although the pairing fails at '
                         'boosted points')


class TestCppHelRecyclingCrossing(unittest.TestCase):
    """`output standalone --hel_recycling=True` (madmatrix) with crossing.

    A directory serves its folded crossings through the extended flavor id,
    and every lane evaluates the BASE helicity row: a leg that changed side
    is fed the opposite helicity with the opposite NSF flag. The recycled
    block (calculate_all_jamps) therefore needs no per-crossing table, but
    its crossed externals must carry that (-nsf, -h) call and the per-lane
    momentum gather. Checked here event by event against the same output
    without recycling, on pages that mix crossings: |M|^2, and the selected
    helicity and colour for fixed random numbers."""

    debugging = getattr(unittest, 'debug', False)
    tolerance = 1e-12
    nevt = 64

    def setUp(self):
        if not shutil.which(os.environ.get('CXX', 'g++')):
            self.skipTest('no C++ compiler available')
        self.cmd = cmd_interface.MasterCmd()
        self.cmd.no_notification()
        self.tmpdir = tempfile.mkdtemp(prefix='cpp_helrecycling_')

    def tearDown(self):
        if not self.debugging and os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _generate(self, process, outputs, defines=()):
        self.cmd.exec_cmd('set automatic_html_opening False')
        self.cmd.exec_cmd('import model sm')
        for define in defines:
            self.cmd.exec_cmd('define %s' % define)
        self.cmd.exec_cmd('generate %s' % process)
        for name, options in outputs:
            self.cmd.exec_cmd('output standalone %s -f %s'
                              % (pjoin(self.tmpdir, name), options))

    def _pdirs(self, name):
        root = pjoin(self.tmpdir, name, 'SubProcesses')
        return [pjoin(root, d) for d in sorted(os.listdir(root))
                if d.startswith('P') and os.path.isdir(pjoin(root, d))]

    def _run(self, pdir, ids, tag, first):
        """Build check_sa.exe (FPTYPE=d) and run it on the given extended ids,
        returning (|M|^2, helicity, colour) per event."""
        import struct
        env = dict(os.environ, FPTYPE='d')
        with open(os.devnull, 'w') as devnull:
            rc = subprocess.call(['make', '-j4', 'check_sa.exe'], cwd=pdir,
                                 stdout=devnull, stderr=subprocess.STDOUT,
                                 env=env)
        self.assertEqual(rc, 0, 'make failed in %s' % pdir)
        files = dict((key, pjoin(self.tmpdir, '%s_%s' % (os.path.basename(pdir), key)))
                     for key in ('flv', 'rh', 'rc', 'mom'))
        if first:
            import random
            rng = random.Random(11)
            with open(files['flv'], 'wb') as fsock:
                fsock.write(struct.pack('<%dI' % self.nevt, *ids))
            for key in ('rh', 'rc'):
                with open(files[key], 'wb') as fsock:
                    fsock.write(struct.pack('<%dd' % self.nevt,
                                            *[rng.random() for _ in range(self.nevt)]))
        out = dict((key, pjoin(self.tmpdir, '%s_%s' % (tag, key)))
                   for key in ('me', 'hel', 'col'))
        command = ['./check_sa.exe', 'perf', '--flavors', files['flv'],
                   '--rnd-hel', files['rh'], '--rnd-col', files['rc'],
                   '--dump-me', out['me'], '--dump-hel', out['hel'],
                   '--dump-col', out['col'],
                   '--dump-momenta' if first else '--momenta', files['mom'],
                   '1', str(self.nevt), '1']
        subprocess.check_output(command, cwd=pdir, stderr=subprocess.STDOUT)

        def read(path, fmt):
            data = open(path, 'rb').read()
            return struct.unpack('<%d%s' % (len(data) // struct.calcsize(fmt), fmt), data)
        return read(out['me'], 'd'), read(out['hel'], 'i'), read(out['col'], 'i')

    def _compare_recycled(self, plain_name, recycled_name):
        """Every crossing row of each directory, mixed in one page: recycled ==
        not recycled event by event. Returns the number of directories that
        fold a crossing."""
        ncrossed = 0
        for plain, recycled in zip(self._pdirs(plain_name), self._pdirs(recycled_name)):
            self.assertEqual(os.path.basename(plain), os.path.basename(recycled))
            calls = pjoin(recycled, 'HelRecyclingCalls.inc')
            self.assertTrue(os.path.exists(calls),
                            'no recycled build in %s' % recycled)
            tables = open(pjoin(recycled, 'ProcessTables.h')).read()
            ncross = int(re.search(r'constexpr int ncross = (\d+);', tables).group(1))
            nflav = int(re.search(r'constexpr int nmaxflavor = (\d+);', open(
                pjoin(recycled, 'ProcessData.h')).read()).group(1))
            if ncross > 1:
                ncrossed += 1
                self.assertIn('icsign', open(calls).read())
            ids = [i % (ncross * nflav) for i in range(self.nevt)]
            me0, hel0, col0 = self._run(plain, ids, 'plain', True)
            me1, hel1, col1 = self._run(recycled, ids, 'recycled', False)
            for i in range(self.nevt):
                self.assertGreater(me0[i], 0)
                self.assertLessEqual(abs(me1[i] - me0[i]), self.tolerance * me0[i],
                                     '%s event %d (id %d): %r vs %r'
                                     % (os.path.basename(plain), i, ids[i],
                                        me1[i], me0[i]))
                self.assertEqual((hel1[i], col1[i]), (hel0[i], col0[i]),
                                 '%s event %d (id %d): helicity/colour'
                                 % (os.path.basename(plain), i, ids[i]))
        return ncrossed

    def test_pp_jj_recycled_crossings(self):
        """p p > j j: every crossing row of each folded directory, mixed in
        one page, recycled == not recycled"""
        self._generate('p p > j j', [('plain', ''),
                                     ('recycled', '--hel_recycling=True')])
        self.assertGreater(self._compare_recycled('plain', 'recycled'), 0,
                           'no directory folds a crossing')

    def test_off_shell_crossing_recycled(self):
        """--crossing_table=all puts the z of p p > z j into the initial
        state, a slot of another mass: the warm-up scans that crossing on
        points of its own masses, so the directory is recycled, and agrees
        with the plain build for every crossing"""
        self._generate('p p > z j', [('plain', '--crossing_table=all'),
                                     ('recycled', '--hel_recycling=True '
                                                  '--crossing_table=all')])
        for pdir in self._pdirs('recycled'):
            tables = open(pjoin(pdir, 'ProcessTables.h')).read()
            onshell = re.search(r'xonshell_tab\[ncross\] = \{([^}]*)\}', tables)
            self.assertTrue(onshell, tables)
            self.assertIn('0', [x.strip() for x in onshell.group(1).split(',')])
        self.assertGreater(self._compare_recycled('plain', 'recycled'), 0,
                           'no directory folds a crossing')

    @staticmethod
    def _tables(pdir):
        """npar, nflav, ncross, the flavor PDGs and the crossing tables a
        madmatrix directory was written with."""
        tables = open(pjoin(pdir, 'ProcessTables.h')).read()
        data = open(pjoin(pdir, 'ProcessData.h')).read()

        def ints(text, name):
            found = re.search(r'\b%s\[[^=]*=\s*\{(.*?)\};' % name, text, re.S)
            return [int(x) for x in re.findall(r'-?\d+', found.group(1))] if found else []
        npar = len(ints(data, 'flavorPDGs')) // int(re.search(
            r'constexpr int nmaxflavor = (\d+);', data).group(1))
        pdgs = ints(data, 'flavorPDGs')
        return {'npar': npar,
                'pdgs': [pdgs[i:i + npar] for i in range(0, len(pdgs), npar)],
                'ncross': int(re.search(r'constexpr int ncross = (\d+);',
                                        tables).group(1)),
                'xpinv': ints(tables, 'xpinv_tab'),
                'xsgni': ints(tables, 'xsgni_tab'),
                'xonshell': ints(tables, 'xonshell_tab')}

    @staticmethod
    def _crossed_pdgs(tables, xid):
        """The PDGs, in input-slot order, of extended flavor id xid."""
        nflav, npar = len(tables['pdgs']), tables['npar']
        cross, base = xid // nflav, tables['pdgs'][xid % nflav]
        if cross == 0:
            return list(base)
        out = []
        for k in range(npar):
            pdg = base[tables['xpinv'][cross * npar + k]]
            if pdg not in (21, 22, 23, 25):
                pdg *= tables['xsgni'][cross * npar + k]
            out.append(pdg)
        return out

    @staticmethod
    def _two_to_two(masses, rng):
        """An on-shell 2 > 2 point at the given masses, boosted along z."""
        m1, m2, m3, m4 = masses
        rs = 1.3 * max(m1 + m2, m3 + m4, 200.) + 1500. * rng.random()
        s = rs * rs
        lam = lambda a, b: math.sqrt((s - (a + b) ** 2) * (s - (a - b) ** 2)) / (2 * rs)
        pin, pout = lam(m1, m2), lam(m3, m4)
        cth, phi = 2 * rng.random() - 1, 2 * math.pi * rng.random()
        sth = math.sqrt(1 - cth * cth)
        v = [pout * sth * math.cos(phi), pout * sth * math.sin(phi), pout * cth]
        point = [[math.hypot(pin, m1), 0., 0., pin],
                 [math.hypot(pin, m2), 0., 0., -pin],
                 [math.hypot(pout, m3)] + v,
                 [math.hypot(pout, m4)] + [-x for x in v]]
        bz = 1.2 * (rng.random() - 0.5)
        gam = 1 / math.sqrt(1 - bz * bz)
        return [[gam * (p[0] + bz * p[3]), p[1], p[2], gam * (p[3] + bz * p[0])]
                for p in point]

    def _run_at(self, pdir, tag, ids, points):
        """Build check_sa.exe (FPTYPE=d) and run it at the given momenta and
        extended ids (at most nevt events, padded to a full page with the
        last), returning (|M|^2, helicity, colour) per given event."""
        import struct
        import random
        nreal = len(ids)
        ids = list(ids) + [ids[-1]] * (self.nevt - nreal)
        points = list(points) + [points[-1]] * (self.nevt - nreal)
        env = dict(os.environ, FPTYPE='d')
        with open(os.devnull, 'w') as devnull:
            rc = subprocess.call(['make', '-j4', 'check_sa.exe'], cwd=pdir,
                                 stdout=devnull, stderr=subprocess.STDOUT,
                                 env=env)
        self.assertEqual(rc, 0, 'make failed in %s' % pdir)
        files = dict((key, pjoin(self.tmpdir, '%s_%s' % (tag, key)))
                     for key in ('flv', 'rh', 'rc', 'mom', 'me', 'hel', 'col'))
        rng = random.Random(11)
        with open(files['flv'], 'wb') as fsock:
            fsock.write(struct.pack('<%dI' % len(ids), *ids))
        for key in ('rh', 'rc'):
            with open(files[key], 'wb') as fsock:
                fsock.write(struct.pack('<%dd' % len(ids),
                                        *[rng.random() for _ in ids]))
        with open(files['mom'], 'wb') as fsock:
            fsock.write(struct.pack('<%dd' % (len(points) * len(points[0]) * 4),
                                    *[x for point in points for p in point for x in p]))
        subprocess.check_output(
            ['./check_sa.exe', 'perf', '--flavors', files['flv'],
             '--rnd-hel', files['rh'], '--rnd-col', files['rc'],
             '--dump-me', files['me'], '--dump-hel', files['hel'],
             '--dump-col', files['col'], '--momenta', files['mom'],
             '1', str(len(ids)), '1'], cwd=pdir, stderr=subprocess.STDOUT)

        def read(path, fmt):
            data = open(path, 'rb').read()
            return struct.unpack('<%d%s' % (len(data) // struct.calcsize(fmt), fmt), data)[:nreal]
        return read(files['me'], 'd'), read(files['hel'], 'i'), read(files['col'], 'i')

    def test_axial_gauge_near_beam_recycled(self):
        """--axial_gauge: d d~ > g g with a gluon within 1e-6 rad of the beam.
        The diagrams that the reference-vector choice zeroes at a helicity are
        zero only for exactly lightlike momenta: their rounding residual, grown
        by the 1/theta of the axial polarisation, is part of the gauge
        cancellation (dropped, the recycled build was 1.5e-5 off at 1e-6 rad).
        The warm-up keeps them: recycled == not recycled"""
        self._generate('d d~ > g g',
                       [('plain', '--axial_gauge=True'),
                        ('recycled', '--hel_recycling=True --axial_gauge=True')])
        points = []
        for theta in (1e-3, 1e-5, 1e-6, math.pi - 1e-6):
            e, phi = 500., 0.3
            v = [e * math.sin(theta) * math.cos(phi),
                 e * math.sin(theta) * math.sin(phi), e * math.cos(theta)]
            points.append([[e, 0., 0., e], [e, 0., 0., -e], [e] + v,
                           [e] + [-x for x in v]])
        ids = [0] * len(points)
        (plain,), (recycled,) = self._pdirs('plain'), self._pdirs('recycled')
        self.assertTrue(os.path.exists(pjoin(recycled, 'HelRecyclingCalls.inc')))
        me0, hel0, col0 = self._run_at(plain, 'plain', ids, points)
        me1, hel1, col1 = self._run_at(recycled, 'recycled', ids, points)
        for i in range(len(points)):
            self.assertGreater(me0[i], 0)
            self.assertLessEqual(abs(me1[i] - me0[i]), self.tolerance * me0[i],
                                 'point %d: %r vs %r' % (i, me1[i], me0[i]))
            self.assertEqual((hel1[i], col1[i]), (hel0[i], col0[i]))

    def test_massive_leg_crossing_of_the_default_mode(self):
        """define q = g t t~; generate q q > q q: the default mode records
        g t > g t (and its kin) on g g > t t~, moving a top into a gluon slot,
        so the good helicities of those crossings are scanned at their own
        masses. Every crossing, at on-shell points of its own masses: recycled
        == not recycled event by event, and both == the expanded output"""
        import random
        self._generate('q q > q q', [('plain', ''),
                                     ('recycled', '--hel_recycling=True'),
                                     ('expanded', '--use_crossing=False')],
                       defines=('q = g t t~',))
        card = open(pjoin(self.tmpdir, 'plain', 'Cards', 'param_card.dat')).read()
        mtop = float(re.search(r'^\s*6\s+(\S+)\s*#\s*MT\b', card, re.M).group(1))
        mass = lambda pdg: mtop if abs(pdg) == 6 else 0.
        refs = {}
        for pdir in self._pdirs('expanded'):
            for flav, pdgs in enumerate(self._tables(pdir)['pdgs']):
                refs[tuple(pdgs)] = (pdir, flav, False)
                refs.setdefault(tuple([pdgs[1], pdgs[0]] + pdgs[2:]),
                                (pdir, flav, True))
        noffshell = 0
        for plain, recycled in zip(self._pdirs('plain'), self._pdirs('recycled')):
            tables = self._tables(plain)
            nids = tables['ncross'] * len(tables['pdgs'])
            if nids == 1:
                continue
            self.assertTrue(os.path.exists(pjoin(recycled, 'HelRecyclingCalls.inc')),
                            'no recycled build in %s' % recycled)
            noffshell += tables['xonshell'].count(0)
            rng = random.Random(5)
            ids = [i % nids for i in range(self.nevt)]
            points = [self._two_to_two([mass(p) for p in
                                        self._crossed_pdgs(tables, xid)], rng)
                      for xid in ids]
            name = os.path.basename(plain)
            me0, hel0, col0 = self._run_at(plain, name + '_plain', ids, points)
            me1, hel1, col1 = self._run_at(recycled, name + '_recycled', ids, points)
            expected = [None] * self.nevt
            byref = {}
            for i, xid in enumerate(ids):
                pdgs = tuple(self._crossed_pdgs(tables, xid))
                self.assertIn(pdgs, refs, 'the expanded output has no %s' % (pdgs,))
                pdir, flav, swap = refs[pdgs]
                point = points[i]
                if swap:
                    point = [point[1], point[0]] + point[2:]
                byref.setdefault(pdir, []).append((i, flav, point))
            for pdir, events in byref.items():
                me, _, _ = self._run_at(pdir, name + '_ref', [e[1] for e in events],
                                        [e[2] for e in events])
                for (i, _, _), value in zip(events, me):
                    expected[i] = value
            for i in range(self.nevt):
                label = '%s event %d (id %d %s)' % (name, i, ids[i],
                                                    self._crossed_pdgs(tables, ids[i]))
                self.assertGreater(expected[i], 0, label)
                self.assertLessEqual(abs(me0[i] - expected[i]), 1e-10 * expected[i],
                                     '%s: folded %r vs expanded %r'
                                     % (label, me0[i], expected[i]))
                self.assertLessEqual(abs(me1[i] - me0[i]), self.tolerance * me0[i],
                                     '%s: recycled %r vs plain %r'
                                     % (label, me1[i], me0[i]))
                self.assertEqual((hel1[i], col1[i]), (hel0[i], col0[i]),
                                 '%s: helicity/colour' % label)
        self.assertGreater(noffshell, 0, 'no recorded crossing moves the top '
                           'into a slot of another mass')
