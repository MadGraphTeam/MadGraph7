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
"""Acceptance tests for the FKS Born building-block standalone output
('output standalone_fortran --fks').

These tests compile and run Fortran (the lightweight 'check_fks' driver), so
they live in the acceptance tier rather than unit_tests (whose 1s budget would
silently skip them). They cover hardcoded Born values, production equivalence,
multi-point determinism, output shape, invalid controls and FKS limits.
"""
from __future__ import division
from __future__ import absolute_import
import glob
import os
import re
import shutil
import subprocess
import tempfile
import unittest

from madgraph import MG5DIR, MadGraph5Error
import madgraph.interface.master_interface as MGCmd
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.various.misc as misc

pjoin = os.path.join


# Reference Born building blocks for each process, evaluated at the phase-space
# point built by the driver's fixed RAMBO seed (RMARIN(1802,9373) in
# check_sa_fks.f). If the driver's phase-space construction or the model
# defaults ever change, regenerate these by running 'check_fks' once and pasting
# the printed values.
#
# The list is the single source of truth for both tests below; adding a process
# here grows the value-regression and the standalone-vs-aMC@NLO cross-check at
# once. Three representative cases are covered:
#   * g g > t t~   : 2 -> 2, gluon legs (non-zero BORNTILDE, full color-link web)
#   * e+ e- > j j  : colour-singlet initial state, a single q-qbar colour link
#   * z > j j      : a 1 -> 2 decay (NINCOMING=1) with a mixed QCD/QED Born,
#                    which exercises the decay kinematics and the
#                    perturbed-split-order selection in the driver
PROCESSES = [
    {'id': 'gg_ttx',
     'process': 'g g > t t~ [QCD]',
     'model': 'loop_sm',
     'born': 0.56845354707929208,
     'borntilde': 0.022972741277183555,
     'bij': {(1, 1): 1.2643814370872453,
             (1, 2): -1.5674461029801903,
             (1, 3): 0.046354897525540478,
             (1, 4): -1.0076716687198406,
             (2, 2): 1.2643814370872453,
             (2, 3): -1.0076716687198406,
             (2, 4): 0.046354897525540478,
             (3, 3): 0.56194730537210891,
             (3, 4): -0.16257783954991759,
             (4, 4): 0.56194730537210891}},
    {'id': 'ee_jj',
     'process': 'e+ e- > j j [QCD]',
     'model': 'loop_sm',
     'born': 0.016492937363039667,
     'borntilde': 0.0,
     'bij': {(3, 3): 0.01630416725597162,
             (3, 4): -0.032608334511943241,
             (4, 4): 0.01630416725597162}},
    {'id': 'z_jj',
     'process': 'z > j j QED=2 [QCD]',
     'model': 'loop_sm',
     'born': 1705.2784047399880,
     'borntilde': 0.0,
     'bij': {(2, 2): 1685.7606208572352,
             (2, 3): -3371.5212417144703,
             (3, 3): 1685.7606208572352}},
    {'id': 'uux_ttx',
     'process': 'u u~ > t t~ [QCD]',
     'model': 'loop_sm',
     'born': 0.59212443086238242,
     'borntilde': 0.0,
     'bij': {(1, 1): 0.5853472637786112,
             (1, 2): 0.14633681594465273,
             (1, 3): -1.0243577116125695,
             (1, 4): -0.29267363188930562,
             (2, 2): 0.5853472637786112,
             (2, 3): -0.29267363188930562,
             (2, 4): -1.0243577116125695,
             (3, 3): 0.58534726377861090,
             (3, 4): 0.14633681594465273,
             (4, 4): 0.58534726377861090}},
    # an EW ([QED]) correction: the soft particle is a photon, so the
    # building blocks are charge-linked Borns (need_charge_links, itype=1).
    # This exercises the whole [QED] path: born_charges.inc, the charge
    # branch of sborn_sf, and the find_color_links(pert='QED') topology
    # (every charged pair, incl. the massive-W self-links). The W is
    # colourless, so its charge links are absent from the colour basis -
    # the value-set here would be wrong if they were ever dropped.
    # The (1,1)/(2,2) entries are the massless-u charge diagonals the driver
    # rebuilds from charge conservation (B_ii = -1/2 sum_{j!=i} B_ij), the
    # same identity used for the massless gluon/quark colour diagonals above.
    {'id': 'uux_wpwm',
     'process': 'u u~ > w+ w- [QED]',
     'model': 'loop_sm',
     # the soft-photon limit test (test_soft_col_limits) fails for this
     # process in a plain aMC@NLO output too (identical failures), so it is
     # kept out of the '--limits' check until that is understood
     'limits': False,
     'born': 0.0053974535211713691,
     'borntilde': 0.0,
     'bij': {(1, 1): 0.00011690478264832319,
             (1, 2): -0.00023380956529664638,
             (1, 3): -0.0003507143479449694,
             (1, 4): 0.0003507143479449694,
             (2, 2): 0.00011690478264832319,
             (2, 3): 0.0003507143479449694,
             (2, 4): -0.0003507143479449694,
             (3, 3): 0.0002630357609587269,
             (3, 4): -0.0005260715219174538,
             (4, 4): 0.0002630357609587269}},
]


class TestFKSStandalone(unittest.TestCase):
    """FKS Fortran standalone end-to-end (generate + launch + check_fks)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='fkssa')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _new_cmd():
        cmd = MGCmd.MasterCmd()
        cmd.no_notification()
        cmd.run_cmd('set automatic_html_opening False --no_save')
        return cmd

    @staticmethod
    def _run(cmd, line):
        # precmd/postcmd=True so the command history is populated (the FKS
        # finalize reads history.get('generate') to parse the perturbation)
        cmd.exec_cmd(line, errorhandling=False, printcmd=False,
                     precmd=True, postcmd=True)

    def _output_fks_sa(self, process, model, path, limits=False):
        """generate + 'output standalone_fortran --fks' + launch (builds & runs
        check_fks, and with limits=True also test_soft_col_limits)."""
        cmd = self._new_cmd()
        self._run(cmd, 'import model %s' % model)
        self._run(cmd, 'generate %s' % process)
        self._run(cmd, 'output standalone_fortran --fks %s%s -f'
                  % ('--limits ' if limits else '', path))
        self._run(cmd, 'launch %s -f' % path)

    def _output_amcatnlo(self, process, model, path):
        """generate + plain (full) aMC@NLO output of the same process."""
        cmd = self._new_cmd()
        self._run(cmd, 'import model %s' % model)
        self._run(cmd, 'generate %s' % process)
        self._run(cmd, 'output %s -f' % path)

    def _born_dir(self, path):
        """the P* subprocess directory holding the standalone driver.

        A multi-flavour process produces one born dir per flavour family
        (e.g. P0_epem_ddx and P0_epem_uux); glob order is filesystem- and
        platform-dependent, so we sort and always take the first one. The
        reference values in PROCESSES are tied to that same deterministic
        choice (the alphabetically-first born dir)."""
        found = sorted(glob.glob(
            pjoin(path, 'SubProcesses', 'P*', 'check_sa_fks.f')))
        self.assertTrue(found, 'no FKS standalone born dir in %s' % path)
        return os.path.dirname(found[0])

    def _run_check_fks(self, born_dir, points=None, seed=(1802, 9373),
                       energy=0, calls=1):
        """run an already-built check_fks and return its stdout."""
        exe = pjoin(born_dir, 'check_fks')
        self.assertTrue(os.path.isfile(exe),
                        'check_fks not built in %s' % born_dir)
        command = [exe]
        if points is not None:
            command.extend([str(energy), str(calls), str(points),
                            str(seed[0]), str(seed[1])])
        out = subprocess.check_output(command, cwd=born_dir,
                                      stderr=subprocess.STDOUT)
        return out.decode(errors='replace')

    @staticmethod
    def _fortran_float(value):
        return float(value.replace('D', 'E').replace('d', 'e'))

    @classmethod
    def parse_check_fks_points(cls, output):
        """Parse the stable POINT/P/value records emitted by check_fks."""
        points = []
        point = None
        for line in output.splitlines():
            toks = line.split()
            if len(toks) == 2 and toks[0] == 'POINT':
                point = {'index': int(toks[1]), 'momenta': {},
                         'born': None, 'borntilde': None, 'bij': {}}
                points.append(point)
            elif point is not None and len(toks) == 6 and toks[0] == 'P':
                point['momenta'][int(toks[1])] = tuple(
                    cls._fortran_float(v) for v in toks[2:])
            elif point is not None and len(toks) == 3 and \
                    toks[0] == 'BORN' and toks[1] == '=':
                point['born'] = cls._fortran_float(toks[2])
            elif point is not None and len(toks) == 3 and \
                    toks[0] == 'BORNTILDE' and toks[1] == '=':
                point['borntilde'] = cls._fortran_float(toks[2])
            elif point is not None and len(toks) == 4 and toks[0] == 'B_ij':
                key = (int(toks[1]), int(toks[2]))
                point['bij'][key] = cls._fortran_float(toks[3])
        return points

    @classmethod
    def parse_check_fks(cls, output):
        """Return the historical tuple interface for the first point."""
        points = cls.parse_check_fks_points(output)
        if not points:
            return None, None, {}
        point = points[0]
        return point['born'], point['borntilde'], point['bij']

    def assertClose(self, value, ref, rel=1e-6, msg=''):
        self.assertIsNotNone(value, 'missing value (%s)' % msg)
        self.assertAlmostEqual(value, ref, delta=abs(ref) * rel + 1e-9,
                               msg='%s: %r vs %r' % (msg, value, ref))

    def assertPointsClose(self, actual, expected, rel=1e-9):
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertEqual(left['index'], right['index'])
            self.assertEqual(left['momenta'], right['momenta'])
            self.assertClose(left['born'], right['born'], rel=rel,
                             msg='point %d BORN' % left['index'])
            self.assertClose(left['borntilde'], right['borntilde'], rel=rel,
                             msg='point %d BORNTILDE' % left['index'])
            self.assertEqual(set(left['bij']), set(right['bij']))
            for key in left['bij']:
                self.assertClose(left['bij'][key], right['bij'][key], rel=rel,
                    msg='point %d B_ij%s' % (left['index'], key))

    # ------------------------------------------------------------------ #
    # tests
    # ------------------------------------------------------------------ #
    def _workdir(self, spec):
        """a fresh per-process working directory under the test tmpdir."""
        path = pjoin(self.tmpdir, spec['id'])
        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    def _check_born_values(self, spec):
        """check_fks reproduces the hardcoded Born building blocks for one
        process at the driver's fixed RAMBO seed."""
        path = pjoin(self._workdir(spec), 'fks_sa')
        self._output_fks_sa(spec['process'], spec['model'], path)
        born_dir = self._born_dir(path)
        points = self.parse_check_fks_points(self._run_check_fks(born_dir))
        self.assertEqual([p['index'] for p in points], [1])
        self.assertEqual(sorted(points[0]['momenta']),
                         list(range(1, len(points[0]['momenta']) + 1)))
        born, borntilde, bij = self.parse_check_fks(
            self._run_check_fks(born_dir))

        # Normal mode retains only the Born-oracle closure.
        self.assertFalse(glob.glob(pjoin(born_dir, 'matrix_*.f')))
        self.assertFalse(glob.glob(pjoin(born_dir, 'V[0-9]*')))
        self.assertFalse(os.path.lexists(pjoin(born_dir,
                                               'MadLoop5_resources')))

        self.assertClose(born, spec['born'], msg='BORN')
        self.assertClose(borntilde, spec['borntilde'], msg='BORNTILDE')
        self.assertEqual(set(bij), set(spec['bij']),
                         'unexpected set of color links')
        for key, ref in spec['bij'].items():
            self.assertClose(bij[key], ref, msg='B_ij%s' % (key,))

    def test_fks_standalone_born_values(self):
        """check_fks reproduces the hardcoded Born building blocks at the
        driver's fixed RAMBO seed, for every process in PROCESSES."""
        for spec in PROCESSES:
            with self.subTest(process=spec['process']):
                self._check_born_values(spec)

    def _check_vs_amcatnlo(self, spec):
        """the standalone Born building blocks match the ones computed by the
        code of a plain aMC@NLO output of the same process.

        The '--fks' exporter first generates a full FKS directory and then
        trims it, so retained born.f / sborn_sf.f / b_sf_*.f must remain
        byte-for-byte production files. We bring the standalone driver into a
        plain aMC@NLO directory, build it there, and require identical values."""
        workdir = self._workdir(spec)
        # 1) standalone_fortran --fks (driver already built by launch)
        path_sa = pjoin(workdir, 'fks_sa')
        self._output_fks_sa(spec['process'], spec['model'], path_sa)
        born_dir_sa = self._born_dir(path_sa)
        val_sa = self.parse_check_fks(self._run_check_fks(born_dir_sa))

        # 2) plain aMC@NLO output of the same process
        path_nlo = pjoin(workdir, 'nlo')
        self._output_amcatnlo(spec['process'], spec['model'], path_nlo)

        # 3) drop the standalone driver + data into the full dir and build it
        pname = os.path.basename(born_dir_sa)
        born_dir_nlo = pjoin(path_nlo, 'SubProcesses', pname)
        production_sources = ['born.f', 'sborn_sf.f'] + [
            os.path.basename(f) for f in glob.glob(
                pjoin(born_dir_sa, 'b_sf_*.f'))]
        for filename in production_sources:
            with open(pjoin(born_dir_sa, filename), 'rb') as sa_file, \
                    open(pjoin(born_dir_nlo, filename), 'rb') as nlo_file:
                self.assertEqual(sa_file.read(), nlo_file.read(), filename)
        for f in ('check_sa_fks.f', 'born_pmass.inc', 'born_links.dat',
                  'born_charges.inc'):
            shutil.copy(pjoin(born_dir_sa, f), pjoin(born_dir_nlo, f))
        # the plain output lacks the makefile-include stubs that the SA
        # finalize writes; the check_fks target does not use their content
        for stub in ('analyse_opts', 'pythia8_opts'):
            sp = pjoin(path_nlo, 'SubProcesses', stub)
            if not os.path.isfile(sp):
                with open(sp, 'w') as fsock:
                    fsock.write('')
        misc.compile(cwd=pjoin(path_nlo, 'Source'))
        misc.compile(['check_fks'], cwd=born_dir_nlo)
        val_nlo = self.parse_check_fks(self._run_check_fks(born_dir_nlo))

        # 4) identical Born building blocks (same code, same seed)
        self.assertClose(val_sa[0], val_nlo[0], rel=1e-9, msg='BORN')
        self.assertClose(val_sa[1], val_nlo[1], rel=1e-9, msg='BORNTILDE')
        self.assertEqual(set(val_sa[2]), set(val_nlo[2]),
                         'color link sets differ between outputs')
        for key in val_sa[2]:
            self.assertClose(val_sa[2][key], val_nlo[2][key], rel=1e-9,
                             msg='B_ij%s' % (key,))

    def _build_production_oracle(self, spec, born_dir_sa, path_nlo):
        """Generate plain aMC@NLO and install the standalone driver/data."""
        self._output_amcatnlo(spec['process'], spec['model'], path_nlo)
        born_dir_nlo = pjoin(path_nlo, 'SubProcesses',
                             os.path.basename(born_dir_sa))
        for filename in ('check_sa_fks.f', 'born_pmass.inc', 'born_links.dat',
                         'born_charges.inc'):
            shutil.copy(pjoin(born_dir_sa, filename),
                        pjoin(born_dir_nlo, filename))
        for stub in ('analyse_opts', 'pythia8_opts'):
            path = pjoin(path_nlo, 'SubProcesses', stub)
            if not os.path.isfile(path):
                with open(path, 'w') as fsock:
                    fsock.write('')
        misc.compile(cwd=pjoin(path_nlo, 'Source'))
        misc.compile(['check_fks'], cwd=born_dir_nlo)
        return born_dir_nlo

    def test_fks_standalone_vs_amcatnlo(self):
        """the standalone Born building blocks match a plain aMC@NLO output of
        the same process, for every process in PROCESSES."""
        for spec in PROCESSES:
            with self.subTest(process=spec['process']):
                self._check_vs_amcatnlo(spec)

    def test_fks_standalone_multi_point(self):
        """Sixteen deterministic points match production for representative
        QCD, QED and decay/mixed-order processes."""
        by_id = dict((spec['id'], spec) for spec in PROCESSES)
        for process_id in ('gg_ttx', 'uux_wpwm', 'z_jj'):
            spec = by_id[process_id]
            with self.subTest(process=spec['process']):
                workdir = self._workdir(spec)
                path_sa = pjoin(workdir, 'fks_sa_multi')
                self._output_fks_sa(spec['process'], spec['model'], path_sa)
                born_dir_sa = self._born_dir(path_sa)
                born_dir_nlo = self._build_production_oracle(
                    spec, born_dir_sa, pjoin(workdir, 'nlo_multi'))

                output_a = self._run_check_fks(
                    born_dir_sa, points=16, seed=(1802, 9373))
                output_b = self._run_check_fks(
                    born_dir_sa, points=16, seed=(1802, 9373))
                output_other = self._run_check_fks(
                    born_dir_sa, points=16, seed=(1234, 5678))
                output_nlo = self._run_check_fks(
                    born_dir_nlo, points=16, seed=(1802, 9373))
                output_repeated = self._run_check_fks(
                    born_dir_sa, points=16, seed=(1802, 9373), calls=3)

                points_a = self.parse_check_fks_points(output_a)
                self.assertEqual(output_a, output_b)
                self.assertEqual(len(points_a), 16)
                self.assertNotEqual(
                    [p['momenta'] for p in points_a],
                    [p['momenta'] for p in
                     self.parse_check_fks_points(output_other)])
                self.assertPointsClose(
                    points_a, self.parse_check_fks_points(output_nlo))
                self.assertPointsClose(
                    points_a, self.parse_check_fks_points(output_repeated))

                if process_id == 'gg_ttx':
                    # Exercise the public named controls as well as the direct
                    # driver comparison above. The launcher itself verifies
                    # that every requested POINT record was produced.
                    cmd = self._new_cmd()
                    self._run(cmd, 'launch %s --points=3 --seed=1234,5678 '
                                   '--timings=2 --nb_run=1 -f' % path_sa)

    def test_fks_standalone_invalid_options(self):
        """Invalid command combinations and sampling controls fail early."""
        cmd = self._new_cmd()
        self._run(cmd, 'import model loop_sm')
        self._run(cmd, 'generate g g > t t~ [QCD]')
        with self.assertRaisesRegex(MadGraph5Error,
                                    '--limits.*only available'):
            self._run(cmd, 'output --limits %s' %
                      pjoin(self.tmpdir, 'bad_limits'))
        with self.assertRaisesRegex(MadGraph5Error, 'Fortran output'):
            self._run(cmd, 'output standalone --fks %s' %
                      pjoin(self.tmpdir, 'bad_standalone'))

        for options, message in [
                ({'points': 0}, '--points'),
                ({'seed': 'broken'}, '--seed'),
                ({'seed': '31329,0'}, '--seed'),
                ({'timings': -1}, '--timings'),
                ({'timings': 1, 'nb_run': 0}, '--nb_run')]:
            with self.subTest(options=options):
                with self.assertRaisesRegex(MadGraph5Error, message):
                    launch_ext.FKSSALauncher(cmd, self.tmpdir, **options)

    def test_nonconverging_limit_sequence_fails_safely(self):
        """checkres/checkres2 stop at their bounds and report failure when
        no point in a 15-element sequence ever approaches the limit."""
        source_path = pjoin(
            MG5DIR, 'Template', 'NLO', 'SubProcesses', 'fks_singular.f')
        with open(source_path) as fsock:
            source = fsock.read()
        first = source.index('      subroutine checkres(')
        second = source.index('      subroutine checkres2(', first)
        end_match = re.search(r'^      end\s*$', source[second:], re.MULTILINE)
        self.assertIsNotNone(end_match)
        routines = source[first:second] + source[
            second:second + end_match.end()] + '\n'

        driver = """      PROGRAM CHECK_NONCONVERGENCE
      IMPLICIT NONE
      INTEGER I,IRET1,IRET2
      REAL*8 XSEC(15),XLIM,WGT(15),WGTL
      REAL*8 XP(15,0:3,21),LXP(0:3,21)
      REAL*8 XLIM2(15),WGTL2(15)
      REAL*8 XP2(15,0:3,5),LXP2(0:3,5)
      DO I=1,15
        XSEC(I)=2D0
        WGT(I)=1D0
        XLIM2(I)=1D0
        WGTL2(I)=1D0
      ENDDO
      XLIM=1D0
      WGTL=1D0
      XP=0D0
      LXP=0D0
      XP2=0D0
      LXP2=0D0
      OPEN(UNIT=77,FILE='fort.77',STATUS='UNKNOWN')
      OPEN(UNIT=78,FILE='fort.78',STATUS='UNKNOWN')
      CALL CHECKRES(XSEC,XLIM,WGT,WGTL,XP,LXP,
     &  0,15,1,4,1,2,IRET1)
      CALL CHECKRES2(XSEC,XLIM2,WGT,WGTL2,XP2,LXP2,
     &  0,15,1,1,2,IRET2)
      IF (IRET1.NE.1 .OR. IRET2.NE.1) STOP 2
      WRITE(*,*) 'NONCONVERGENCE SAFELY FAILED'
      END

      SUBROUTINE XPRINTOUT(IUNIT,XV,XLIM)
      IMPLICIT NONE
      INTEGER IUNIT
      REAL*8 XV,XLIM
      WRITE(IUNIT,*) XV,XLIM
      END
"""
        test_dir = pjoin(self.tmpdir, 'checkres_bounds')
        os.makedirs(test_dir)
        with open(pjoin(test_dir, 'nexternal.inc'), 'w') as fsock:
            fsock.write('      INTEGER NEXTERNAL\n'
                        '      PARAMETER (NEXTERNAL=4)\n')
        test_source = pjoin(test_dir, 'check_nonconvergence.f')
        with open(test_source, 'w') as fsock:
            fsock.write(driver + routines)
        subprocess.check_call(
            ['gfortran', '-O0', '-fcheck=bounds',
             '-ffixed-line-length-none', '-o',
             'check_nonconvergence', 'check_nonconvergence.f'], cwd=test_dir)
        output = subprocess.check_output(
            ['./check_nonconvergence'], cwd=test_dir,
            stderr=subprocess.STDOUT).decode(errors='replace')
        self.assertIn('NONCONVERGENCE SAFELY FAILED', output)

    def _check_limits(self, spec):
        """'--limits' leaves the Born building blocks unchanged, and the
        soft/collinear limit test that launch runs passes in every born dir.

        The counterterms that test_soft_col_limits checks against the real
        emission are built from born.f / sborn_sf.f / b_sf_*.f, i.e. from
        the very building blocks check_fks prints."""
        path = pjoin(self._workdir(spec), 'fks_sa_limits')
        # launch raises if any born dir fails the limit test
        self._output_fks_sa(spec['process'], spec['model'], path, limits=True)
        born = self.parse_check_fks(
            self._run_check_fks(self._born_dir(path)))[0]
        self.assertClose(born, spec['born'], msg='BORN')
        self._assert_limits_passed(path)

    def _assert_limits_passed(self, path):
        """every born dir ran test_soft_col_limits and no check FAILED;
        returns the born dirs checked."""
        born_dirs = sorted(os.path.dirname(f) for f in glob.glob(
            pjoin(path, 'SubProcesses', 'P*', 'check_sa_fks.f')))
        self.assertTrue(born_dirs, 'no FKS standalone born dir in %s' % path)
        for born_dir in born_dirs:
            log = pjoin(born_dir, 'test_ME.log')
            self.assertTrue(os.path.isfile(log),
                            'limit test not run in %s' % born_dir)
            with open(log, errors='replace') as fsock:
                content = fsock.read()
            self.assertNotIn('FAILED', content, 'limit test failed: %s' % log)
            self.assertNotIn('fixed shat', content,
                             'limit test not run in %s' % born_dir)
            self.assertIn('PASSED', content, 'no limit check in %s' % log)
            self.assertTrue(glob.glob(pjoin(born_dir, 'matrix_*.f')),
                            'real matrix elements missing in %s' % born_dir)
            self.assertFalse(glob.glob(pjoin(born_dir, 'V[0-9]*')),
                             'virtual directory retained in %s' % born_dir)
            self.assertFalse(os.path.lexists(
                pjoin(born_dir, 'MadLoop5_resources')),
                'MadLoop resources retained in %s' % born_dir)
        return born_dirs

    def test_fks_standalone_limits(self):
        """'output standalone_fortran --fks --limits' + launch runs the
        soft/collinear limit test and it passes, for every process in
        PROCESSES that supports it."""
        for spec in PROCESSES:
            if not spec.get('limits', True):
                continue
            with self.subTest(process=spec['process']):
                self._check_limits(spec)

    def test_fks_standalone_limits_pp_ttx(self):
        """soft and collinear limits of p p > t t~ [QCD] through the
        '--limits' standalone output: every partonic channel (gg, q q~,
        q~ q) must pass test_soft_col_limits, the initial-state collinear
        ones included (hadronic beams with the built-in PDF set)."""
        path = pjoin(self.tmpdir, 'pp_ttx_limits')
        self._output_fks_sa('p p > t t~ [QCD]', 'loop_sm', path, limits=True)
        born_dirs = self._assert_limits_passed(path)
        names = [os.path.basename(d) for d in born_dirs]
        self.assertIn('P0_gg_ttx', names)
        self.assertTrue(any(n.startswith('P0_uux') for n in names), names)


if __name__ == '__main__':
    unittest.main()
