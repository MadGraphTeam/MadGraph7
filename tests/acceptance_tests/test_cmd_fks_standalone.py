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
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from madgraph import MG5DIR, MadGraph5Error
import madgraph.interface.master_interface as MGCmd
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.various.banner as banner_mod
from madgraph.iolibs import born_support
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
     # This fixed-parton process has lpp1=lpp2=0. Once IDUP_D correctly
     # contains physical quarks, genps_fks rejects its incoming-j collinear
     # configurations because fixed shat is unsupported. The corresponding
     # hadronic p p > t t~ limit oracle below covers those configurations with
     # PDFs and requires every check to pass.
     'limits': False,
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
     # Exercise physical closed-loop specialization with grouped QED.
     'grouping': True,
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


# processes generated with 'set apply_flavor_grouping False'
UNGROUPED_PROCESSES = set(spec['process'] for spec in PROCESSES
                          if not spec.get('grouping', True))
# The upstream coupling-cache regression exercises the scalar string-coupling
# capability. Grouped caller-owned amplitude snapshots have separate coverage.
UNGROUPED_PROCESSES.add('u u~ > t t~ QCD^2<=4 QED^2<=4 [real=QCD]')


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
        cmd.run_cmd('set nb_core 2 --no_save')
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
        cmd = self._generate_fks_sa(process, model, path, limits=limits)
        self._run(cmd, 'launch %s -f' % path)

    def _set_grouping(self, cmd, process):
        cmd._experimental_nlo_flavor_grouping = process not in UNGROUPED_PROCESSES
        self._run(cmd, 'set apply_flavor_grouping %s --no_save' %
                  (process not in UNGROUPED_PROCESSES))

    def _generate_fks_sa(self, process, model, path, limits=False):
        """Generate FKS standalone source without compiling or launching it."""
        cmd = self._new_cmd()
        self._set_grouping(cmd, process)
        self._run(cmd, 'import model %s' % model)
        self._run(cmd, 'generate %s' % process)
        self._run(cmd, 'output standalone_fortran --fks %s%s -f'
                  % ('--limits ' if limits else '', path))
        return cmd

    def _output_amcatnlo(self, process, model, path):
        """generate + plain (full) aMC@NLO output of the same process."""
        cmd = self._new_cmd()
        self._set_grouping(cmd, process)
        self._run(cmd, 'import model %s' % model)
        self._run(cmd, 'generate %s' % process)
        self._run(cmd, 'output %s -f' % path)
        self._prepare_matrix_probe_cards(path)

    @staticmethod
    def _prepare_matrix_probe_cards(path):
        # The production comparison needs model and helicity libraries but no
        # PDF values. Keep the acceptance test self-contained on systems where
        # LHAPDF is not installed, just like the standalone output under test.
        for name in ('run_card_default.dat', 'run_card.dat'):
            card_path = pjoin(path, 'Cards', name)
            run_card = banner_mod.RunCardNLO(card_path)
            run_card['pdlabel'] = 'nn23nlo'
            run_card['reweight_pdf'] = [False]
            run_card.write(card_path)

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
                       energy=0, calls=1, flavor=0):
        """run an already-built check_fks and return its stdout."""
        exe = pjoin(born_dir, 'check_fks')
        self.assertTrue(os.path.isfile(exe),
                        'check_fks not built in %s' % born_dir)
        command = [exe]
        if points is not None or flavor:
            points = 1 if points is None else points
            command.extend([str(energy), str(calls), str(points),
                            str(seed[0]), str(seed[1]), str(flavor)])
        out = subprocess.check_output(command, cwd=born_dir,
                                      stderr=subprocess.STDOUT)
        return out.decode(errors='replace')

    @staticmethod
    def _native_probe_links(output_path, subproc):
        """Use the real native context/model-state closure for isolated probes."""
        from tests.acceptance_tests.test_nlo_real_offload_contract import TestNLORealFortranOracle
        probe = TestNLORealFortranOracle()
        return (probe._native_real_link_objects(output_path, subproc),
                probe._native_real_link_flags(output_path, subproc))

    @staticmethod
    def _canonicalize_flv_identifiers(source):
        """Normalize process-global generated flavour-helper identifiers.

        ALOHA allocates the numeric suffix in ``FLV_<number>`` from a
        process-global counter.  Two independently generated outputs can
        therefore contain identical source modulo those suffixes.  Preserve
        the identity and reuse pattern of every helper while comparing the
        production sources, rather than collapsing all helpers to one token.
        """
        identifiers = {}

        def replace(match):
            identifier = match.group(0)
            if identifier not in identifiers:
                identifiers[identifier] = len(identifiers) + 1
            return b'FLV_CANONICAL_%d' % identifiers[identifier]

        return re.sub(br'\bFLV_[0-9]+\b', replace, source)

    @staticmethod
    def _fortran_float(value):
        return float(value.replace('D', 'E').replace('d', 'e'))

    @classmethod
    def parse_check_fks_points(cls, output):
        """Parse the stable POINT/P/value records emitted by check_fks."""
        points = []
        point = None
        configuration = None
        for line in output.splitlines():
            toks = line.split()
            if len(toks) == 2 and toks[0] == 'POINT':
                point = {'index': int(toks[1]), 'momenta': {},
                          'born': None, 'borntilde': None, 'bij': {},
                          'configurations': []}
                configuration = None
                points.append(point)
            elif point is not None and len(toks) == 5 and \
                    toks[:3] == ['====', 'FLAVOUR', 'CONFIGURATION'] and \
                    toks[4] == '====':
                configuration = {
                     'index': int(toks[3]), 'born': None,
                     'borntilde': None, 'bij': {}, 'pdgs': None,
                     'flavor_index': None}
                point['configurations'].append(configuration)
            elif configuration is not None and toks and toks[0] == 'PDG':
                configuration['pdgs'] = tuple(int(value)
                                              for value in toks[1:])
            elif configuration is not None and len(toks) == 4 and \
                    toks[:3] == ['LOCAL', 'FLAVOUR', 'INDEX']:
                configuration['flavor_index'] = int(toks[3])
            elif point is not None and len(toks) == 6 and toks[0] == 'P':
                point['momenta'][int(toks[1])] = tuple(
                    cls._fortran_float(v) for v in toks[2:])
            elif point is not None and len(toks) == 3 and \
                    toks[0] == 'BORN' and toks[1] == '=':
                target = configuration if configuration is not None else point
                target['born'] = cls._fortran_float(toks[2])
            elif point is not None and len(toks) == 3 and \
                    toks[0] == 'BORNTILDE' and toks[1] == '=':
                target = configuration if configuration is not None else point
                target['borntilde'] = cls._fortran_float(toks[2])
            elif point is not None and len(toks) == 4 and toks[0] == 'B_ij':
                key = (int(toks[1]), int(toks[2]))
                target = configuration if configuration is not None else point
                target['bij'][key] = cls._fortran_float(toks[3])
        # Preserve the historical top-level interface: for grouped output it
        # denotes the first physical configuration rather than whichever row
        # happened to be printed last.
        for parsed_point in points:
            if parsed_point['configurations']:
                first = parsed_point['configurations'][0]
                parsed_point['born'] = first['born']
                parsed_point['borntilde'] = first['borntilde']
                parsed_point['bij'] = first['bij']
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

    def _check_born_values(self, spec, native=False):
        """check_fks reproduces the hardcoded Born building blocks for one
        process at the driver's fixed RAMBO seed."""
        path = pjoin(self._workdir(spec), 'fks_sa')
        if native:
            self._generate_fks_sa(spec['process'], spec['model'], path, limits=True)
            misc.compile(cwd=pjoin(path, 'Source'))
            misc.compile(['check_fks'], cwd=self._born_dir(path))
        else:
            self._output_fks_sa(spec['process'], spec['model'], path)
        born_dir = self._born_dir(path)
        points = self.parse_check_fks_points(self._run_check_fks(born_dir))
        self.assertEqual([p['index'] for p in points], [1])
        self.assertEqual(sorted(points[0]['momenta']),
                         list(range(1, len(points[0]['momenta']) + 1)))
        born, borntilde, bij = self.parse_check_fks(
            self._run_check_fks(born_dir))

        # Normal mode retains only the Born-oracle closure.
        self.assertEqual(bool(glob.glob(pjoin(born_dir, 'matrix_*.f'))), native)
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

    def test_fks_native_provider_born_values(self):
        """The native Born facade agrees with the same frozen building blocks.

        Keep real source to exercise provider finalization, but only build and
        execute check_fks; no soft-limit scan or integration is launched.
        """
        for spec in PROCESSES:
            with self.subTest(process=spec['process']):
                self._check_born_values(spec, native=True)

    def test_fks_standalone_grouped_physical_flavors(self):
        self._check_grouped_physical_flavors()

    def test_fks_native_grouped_physical_flavors(self):
        self._check_grouped_physical_flavors(native=True)

    def _check_grouped_physical_flavors(self, native=False):
        """Grouped QCD output evaluates every physical Born row independently.

        Two points and three repeated calls exercise the flavor-aware Born and
        color-link caches. Every row is then selected on its own at sixteen
        points and compared with an independent pre-grouping oracle from
        commit 844829d3ef. Down/up rows differ, while generation-equivalent
        first/second-generation rows agree and match the old physical
        subprocesses.
        """

        path = pjoin(self.tmpdir, 'grouped_qcd')
        if native:
            self._generate_fks_sa('p p > w+ w- [real=QCD]', 'loop_sm', path, limits=True)
            misc.compile(cwd=pjoin(path, 'Source'))
            misc.compile(['check_fks'], cwd=self._born_dir(path))
        else:
            self._output_fks_sa('p p > w+ w- [real=QCD]', 'loop_sm', path)
        oracle_path = pjoin(
            MG5DIR, 'tests', 'input_files',
            'nlo_pre_grouping_wpwm_oracle.json')
        with open(oracle_path) as stream:
            pre_grouping = json.load(stream)
        self.assertEqual(
            pre_grouping['metadata']['source_commit'],
            '844829d3ef0b13d294045f34dbbeef3a3d743e9b')
        self.assertEqual(pre_grouping['born']['points'], 16)
        self.assertEqual(pre_grouping['born']['seed'], [1802, 9373])
        expected = {
            'P0_QQx_wpwm': ([(1, -1, 24, -24),
                              (2, -2, 24, -24),
                              (3, -3, 24, -24),
                              (4, -4, 24, -24)], [
                [0.022213760524088497, 0.0051099582274946693],
                [0.0057460860777143481, 0.016065806963586577],
            ]),
            'P0_QxQ_wpwm': ([(-1, 1, 24, -24),
                              (-2, 2, 24, -24),
                              (-3, 3, 24, -24),
                              (-4, 4, 24, -24)], [
                [0.0046865329655036867, 0.022651095898120144],
                [0.015592504815818439, 0.0062100269962616240],
            ]),
        }
        born_dirs = sorted(glob.glob(pjoin(path, 'SubProcesses', 'P*')))
        self.assertEqual(set(os.path.basename(item) for item in born_dirs),
                         set(expected))
        for born_dir in born_dirs:
            self._instrument_physical_flavour_cache(born_dir)
            points = self.parse_check_fks_points(self._run_check_fks(
                born_dir, points=2, seed=(1802, 9373), calls=3))
            self.assertEqual(len(points), 2)
            expected_pdgs, references = expected[os.path.basename(born_dir)]
            for point, reference in zip(points, references):
                configurations = point['configurations']
                self.assertEqual([item['index'] for item in configurations],
                                 [1, 2, 3, 4])
                self.assertEqual([item['pdgs'] for item in configurations],
                                 expected_pdgs)
                self.assertEqual(
                    [item['flavor_index'] for item in configurations],
                    [1, 2, 3, 4])
                self.assertClose(configurations[0]['born'], reference[0])
                self.assertClose(configurations[1]['born'], reference[1])
                self.assertClose(configurations[2]['born'], reference[0])
                self.assertClose(configurations[3]['born'], reference[1])
                self.assertNotEqual(configurations[0]['born'],
                                    configurations[1]['born'])
                self.assertEqual(configurations[0]['bij'],
                                 configurations[2]['bij'])
                self.assertEqual(configurations[1]['bij'],
                                 configurations[3]['bij'])

            # As in the ordinary standalone/MadMatrix driver, a one-based
            # physical flavour selector evaluates only that row. Its first two
            # points must be identical to the same row in the all-flavour run.
            # All sixteen points must also reproduce the standalone output
            # generated by the genuinely pre-grouping source commit above.
            for flavor in range(1, 5):
                selected = self.parse_check_fks_points(self._run_check_fks(
                    born_dir, points=16, seed=(1802, 9373), calls=3,
                    flavor=flavor))
                self.assertEqual(
                    [item['momenta'] for item in selected[:2]],
                    [item['momenta'] for item in points])
                for selected_point, all_point in zip(selected[:2], points):
                    self.assertEqual(len(selected_point['configurations']), 1)
                    actual = selected_point['configurations'][0]
                    reference = all_point['configurations'][flavor - 1]
                    self.assertEqual(actual, reference)

                pdgs = selected[0]['configurations'][0]['pdgs']
                channel_key = ','.join(str(pdg) for pdg in pdgs)
                old_subprocess = pre_grouping['born'][
                    'physical_channels'][channel_key]
                old_points = pre_grouping['born']['oracles'][old_subprocess]
                self.assertEqual(len(old_points), len(selected))
                for index, (actual_point, old_point) in enumerate(
                        zip(selected, old_points)):
                    expected_momenta = dict(
                        (leg, tuple(momentum)) for leg, momentum in enumerate(
                            pre_grouping['born']['momenta'][index], 1))
                    self.assertEqual(actual_point['momenta'],
                                     expected_momenta)
                    self.assertEqual(len(actual_point['configurations']), 1)
                    actual = actual_point['configurations'][0]
                    self.assertEqual(actual['pdgs'], pdgs)
                    for key in ('born', 'borntilde'):
                        expected_value = old_point[key]
                        self.assertAlmostEqual(
                            actual[key], expected_value,
                            delta=max(abs(expected_value) * 1e-12, 1e-14),
                            msg='%s point %d %s' %
                                (channel_key, index + 1, key))
                    expected_bij = dict(
                        (tuple(int(value) for value in key.split(',')), weight)
                        for key, weight in old_point['bij'].items())
                    self.assertEqual(set(actual['bij']), set(expected_bij))
                    for key, expected_value in expected_bij.items():
                        self.assertAlmostEqual(
                            actual['bij'][key], expected_value,
                            delta=max(abs(expected_value) * 1e-12, 1e-14),
                            msg='%s point %d B_ij%s' %
                                (channel_key, index + 1, key))

        # Exercise the public named option as well as the direct positional
        # driver interface used above.
        cmd = self._new_cmd()
        self._run(cmd, 'launch %s --flavor=2 --points=2 -f' % path)

        invalid = subprocess.run(
            [pjoin(born_dirs[0], 'check_fks'),
             '0', '1', '1', '1802', '9373', '5'],
            cwd=born_dirs[0], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn('flavour index must be between 0 and', invalid.stdout)

    @staticmethod
    def _wpwm_real_driver_source():
        """Fixed-point real-ME oracle for grouped W-pair output."""
        return r"""      PROGRAM CHECK_REAL_ORACLE
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'nFKSconfigs.inc'
      INCLUDE 'orders.inc'
      INTEGER I,J,NFKSPROCESS
      DOUBLE PRECISION P(0:3,NEXTERNAL),WGT,MW,PZ
      DOUBLE PRECISION RET_AMP_SPLIT(AMP_SPLIT_SIZE)
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      INCLUDE 'fks_info.inc'
      CALL SETPARA('param_card.dat')
      MW=80.419D0
      PZ=SQRT(400D0**2-100D0**2-MW**2)
      P(:,:)=0D0
      P(0,1)=500D0
      P(3,1)=500D0
      P(0,2)=500D0
      P(3,2)=-500D0
      P(0,3)=400D0
      P(1,3)=-100D0
      P(3,3)=PZ
      P(0,4)=400D0
      P(1,4)=-100D0
      P(3,4)=-PZ
      P(0,5)=200D0
      P(1,5)=200D0
      DO I=1,FKS_CONFIGS
        NFKSPROCESS=I
        CALL SMATRIX_REAL(P,RET_AMP_SPLIT,WGT)
        WRITE(*,*) 'ORACLE_REAL',I,
     $    (PDG_TYPE_D(I,J),J=1,NEXTERNAL),EXTRA_CNT_D(I),WGT
      ENDDO
      END
"""

    def _run_wpwm_real_driver(self, output_path):
        """Build the W-pair real oracle and return all physical FKS rows."""
        misc.compile(cwd=pjoin(output_path, 'Source'))
        records = []
        for subproc in sorted(glob.glob(
                pjoin(output_path, 'SubProcesses', 'P*'))):
            matrix_objects = [
                os.path.basename(path)[:-2] + '.o' for path in
                sorted(glob.glob(pjoin(subproc, 'matrix_*.f')))]
            self.assertTrue(matrix_objects)
            objects = matrix_objects + [
                'real_me_chooser.o', 'splitorders_stuff.o',
                'check_real_oracle.o']
            native_objects, native_flags = self._native_probe_links(output_path, subproc)
            objects += native_objects
            source = pjoin(subproc, 'check_real_oracle.f')
            with open(source, 'w') as stream:
                stream.write(self._wpwm_real_driver_source())
            misc.compile(objects, cwd=subproc)
            executable = pjoin(subproc, 'check_real_oracle')
            subprocess.check_call(
                ['gfortran', '-o', executable] + objects + [
                    '-L%s' % pjoin(output_path, 'lib'),
                    '-ldhelas', '-lmodel'] + native_flags, cwd=subproc)
            output = subprocess.check_output(
                [executable], cwd=subproc, stderr=subprocess.STDOUT)
            for line in output.decode(errors='replace').splitlines():
                tokens = line.split()
                if not tokens or tokens[0] != 'ORACLE_REAL':
                    continue
                records.append({
                    'subprocess': os.path.basename(subproc),
                    'configuration': int(tokens[1]),
                    'pdgs': tuple(int(value) for value in tokens[2:-2]),
                    'extra_counterterm': int(tokens[-2]),
                    'value': self._fortran_float(tokens[-1])})
        return records

    def test_fks_standalone_grouped_real_vs_pre_grouping(self):
        """Every grouped W-pair real row matches commit 844829d3ef.

        ``--limits`` retains the production real matrix elements in FKS
        standalone output. Evaluate all 32 physical FKS classes at one fixed
        point and compare with the independently generated pre-grouping
        subprocesses. This process has no extra counterterms.
        """
        path = pjoin(self.tmpdir, 'grouped_qcd_real')
        self._generate_fks_sa(
            'p p > w+ w- [real=QCD]', 'loop_sm', path, limits=True)
        records = self._run_wpwm_real_driver(path)
        self.assertEqual(len(records), 32)

        oracle_path = pjoin(
            MG5DIR, 'tests', 'input_files',
            'nlo_pre_grouping_wpwm_oracle.json')
        with open(oracle_path) as stream:
            pre_grouping = json.load(stream)
        self.assertEqual(
            pre_grouping['metadata']['source_commit'],
            '844829d3ef0b13d294045f34dbbeef3a3d743e9b')
        old_reals = pre_grouping['real']['oracles']

        def canonical_pdgs(pdgs):
            canonical = []
            for pdg in pdgs:
                sign = -1 if pdg < 0 else 1
                absolute = {3: 1, 4: 2}.get(abs(pdg), abs(pdg))
                canonical.append(sign * absolute)
            return tuple(canonical)

        for record in records:
            self.assertEqual(record['extra_counterterm'], 0)
            key = ','.join(str(pdg) for pdg in
                           canonical_pdgs(record['pdgs']))
            expected = old_reals[key]
            self.assertAlmostEqual(
                record['value'], expected,
                delta=max(abs(expected) * 1e-12, 1e-20),
                msg='%s FKS row %d %s' %
                    (record['subprocess'], record['configuration'],
                     record['pdgs']))

    @staticmethod
    def _instrument_physical_flavour_cache(born_dir):
        """Probe A -> B -> A without manual invalidation in the real Born."""
        path = pjoin(born_dir, 'check_sa_fks.f')
        with open(path) as stream:
            driver = stream.read()
        needle = '        CALL GET_MOMENTA(SQRTS,PMASS,P)\n'
        assert driver.count(needle) == 1
        driver = driver.replace(needle, needle +
                                '        CALL TEST_FLAVOUR_CACHE(P)\n', 1)
        driver += '''
      subroutine test_flavour_cache(p)
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'born_nhel.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      include 'orders.inc'
      double precision p(0:3,nexternal-1),a,b,again,link_a,link_again
      double precision amp2(ngraphs),jamp2(0:ncolor),ret(amp_split_size)
      double precision ref(amp_split_size),scale,worst
      double complex cnt(2,nsplitorders)
      double complex retcnt(amp_split_size,2,nsplitorders)
      double complex saved_a(ngraphs,max_bhel),saved_b(ngraphs,max_bhel)
      double complex original_a(ngraphs,max_bhel)
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer nfksprocess,previous
      common/c_nfksprocess/nfksprocess
      if (NBORN_FLAVOR_CONFIGS.lt.2) stop 94
      previous=nfksprocess
      nfksprocess=BORN_FKS_CONFIG_D(1)
      calculatedBorn=.false.
      call sborn_amp(p,amp2,jamp2,ret,retcnt,a,cnt,saved_a)
      ref=ret
      original_a=saved_a
      call sb_sf_001(p,link_a,retcnt,saved_a)
c Keep CALCULATEDBORN set: production must reject the other physical row.
      nfksprocess=BORN_FKS_CONFIG_D(2)
      call sborn_amp(p,amp2,jamp2,ret,retcnt,b,cnt,saved_b)
      if (a.eq.b) stop 95
      if (any(saved_a.ne.original_a)) stop 96
c Consume A while the global cache still holds B.
      nfksprocess=BORN_FKS_CONFIG_D(1)
      call sb_sf_001(p,link_again,retcnt,saved_a)
      worst=abs(link_a-link_again)/max(abs(link_a),1d-300)
      call sborn_amp(p,amp2,jamp2,ret,retcnt,again,cnt,saved_b)
      scale=max(maxval(abs(ref)),1d-300)
      worst=max(worst,maxval(abs(ret-ref))/scale)
      worst=max(worst,abs(a-again)/max(abs(a),1d-300))
      write(*,*) 'PHYSICAL FLAVOUR CACHE WORST',worst
      if (worst.gt.1d-12) stop 97
      nfksprocess=previous
      calculatedBorn=.false.
      end
'''
        with open(path, 'w') as stream:
            stream.write(driver)
        misc.compile(['check_fks'], cwd=born_dir)

    # Point 1 of the real-emission throughput benchmark (RAMBO at sqrt(s) =
    # 1 TeV, seed 12345, momenta in the order of g g > g g g g) and |M|^2 of
    # the real g g > g g g g there from MG5aMC v3.8.0, at alpha_s = 0.118.
    GGGGGG_POINT = (
        (5.00000000000000000e+02, 0.0, 0.0, 5.00000000000000000e+02),
        (5.00000000000000000e+02, 0.0, 0.0, -5.00000000000000000e+02),
        (2.71415971622457903e+02, 2.29831399887524555e+02,
         -1.38894948452979037e+02, 3.93998803519499958e+01),
        (2.07563041407746653e+02, 1.46691345098497493e+02,
         1.46680505580990911e+02, -6.99247553676969247e+00),
        (2.66799015476846705e+02, -1.81953907107772324e+02,
         -1.00153487182916479e+02, -1.67462740192355142e+02),
        (2.54221971492948654e+02, -1.94568837878249724e+02,
         9.23679300549046474e+01, 1.35055335377174828e+02))
    GGGGGG_MG5AMC_380 = 3.2818692843658304e-06

    @staticmethod
    def _real_cf_driver_source():
        """The real of one FKS configuration at one point, through the scalar
        and through the batched (one lane) production entry points."""
        return r"""      PROGRAM CHECK_REAL_CF
      USE COUPLINGS
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      INCLUDE 'nFKSconfigs.inc'
      INCLUDE 'coupl.inc'
      INTEGER I,J,NFKSPROCESS
      DOUBLE PRECISION P(0:3,NEXTERNAL),WGT
      DOUBLE PRECISION RET_AMP_SPLIT(AMP_SPLIT_SIZE)
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      INCLUDE 'fks_info.inc'
      CALL SETPARA('param_card.dat')
C     alpha_s = 0.118 computed as SETPARA does, whatever the card says
      G=2D0*DSQRT(0.118D0*3.141592653589793D0)
      CALL UPDATE_AS_PARAM()
      CALL ALLOCATE_COUPLINGS(1)
      CALL UPDATE_AS_PARAM_VEC(1)
      READ(*,*) NFKSPROCESS
      DO J=1,NEXTERNAL
        READ(*,*) (P(I,J),I=0,3)
      ENDDO
C     the scalar call also sets up the split-order selection that the
C     batched one reuses, as set_goodhel does in a run
      CALL SMATRIX_REAL(P,RET_AMP_SPLIT,WGT)
      WRITE(*,*) 'REAL_SCALAR',WGT
      CALL SMATRIX_REAL_VEC(P,RET_AMP_SPLIT,WGT,1,NFKSPROCESS,
     $     REAL_FLAVOR_INDEX_D(NFKSPROCESS))
      WRITE(*,*) 'REAL_BATCHED',WGT
      END
"""

    @staticmethod
    def _real_me_number(subproc, process):
        """N of the matrix_N.f of a P directory computing the real `process`,
        or None."""
        for path in sorted(glob.glob(pjoin(subproc, 'matrix_*.f'))):
            with open(path) as stream:
                match = re.search(r'^C\s+Process:\s+(.*?)\s*\[',
                                  stream.read(), re.M)
            if match and match.group(1).split() == process.split():
                return int(re.search(r'matrix_(\d+)\.f$', path).group(1))
        return None

    @staticmethod
    def _color_matrix_entries(source):
        """{routine: (NCOLOR, indices of CF given a value by its DATA
        statements)} for every routine of a matrix_N.f holding a colour
        matrix."""
        entries = {}
        for unit in re.split(r'\n(?=\s{6}(?:RECURSIVE\s+)?SUBROUTINE\s)',
                             source):
            name = re.match(r'\s*(?:RECURSIVE\s+)?SUBROUTINE\s+(\w+)', unit)
            ncolor = re.search(r'\bNCOLOR=(\d+)', unit)
            if not name or not ncolor or \
                    not re.search(r'^\s+INTEGER\s+CF\(', unit, re.M):
                continue
            indices = set()
            for first, last in re.findall(
                    r'DATA\s*\(CF\(I\),\s*I=\s*(\d+)\s*,\s*(\d+)\s*\)', unit):
                indices.update(range(int(first), int(last) + 1))
            entries[name.group(1)] = (int(ncolor.group(1)), indices)
        return entries

    def test_fks_real_color_matrix_gg_gggg(self):
        """The real g g > g g g g of g g > g g g [real=QCD] carries its colour
        matrix and gives the value of MG5aMC v3.8.0.

        Its 120 colour flows are above the size from which
        get_color_matrix_encoding compresses the colour matrix, to be rebuilt
        at run time by INIT_CF. The FKS templates have no INIT_CF: MATRIX_<N>
        and MATRIX_<N>_VEC were written with the denominator alone, CF stayed
        empty and the real was exactly zero at every point, in the scalar and
        in the batched path alike.
        """
        path = pjoin(self.tmpdir, 'gg_ggg_real')
        cmd = self._new_cmd()
        self._run(cmd, 'import model sm')
        self._run(cmd, 'generate g g > g g g [real=QCD]')
        self._run(cmd, 'output %s -f' % path)

        found = [(subproc, self._real_me_number(subproc, 'g g > g g g g'))
                 for subproc in sorted(glob.glob(
                     pjoin(path, 'SubProcesses', 'P*')))]
        found = [(subproc, number) for subproc, number in found if number]
        self.assertEqual(len(found), 1, found)
        subproc, number = found[0]

        with open(pjoin(subproc, 'matrix_%d.f' % number)) as stream:
            entries = self._color_matrix_entries(stream.read())
        self.assertEqual(sorted(entries),
                         ['MATRIX_%d' % number, 'MATRIX_%d_VEC' % number])
        for routine, (ncolor, indices) in entries.items():
            self.assertEqual(ncolor, 120, routine)
            size = ncolor * (ncolor + 1) // 2
            self.assertEqual(indices, set(range(1, size + 1)),
                             '%s: %d of the %d colour matrix entries written'
                             % (routine, len(indices), size))

        # an FKS configuration whose real emission this is
        with open(pjoin(subproc, 'real_me_chooser.f')) as stream:
            scalar = stream.read().split(
                'RECURSIVE SUBROUTINE SMATRIX_REAL_VEC', 1)[0]
        rows = [int(row) for row, real in re.findall(
                    r'NFKSPROCESS\.EQ\.(\d+)\)\s*THEN\s*'
                    r'CALL\s+SMATRIX(\d+)_AMP\(', scalar, re.I)
                if int(real) == number]
        self.assertTrue(rows, 'no FKS configuration calls SMATRIX%d' % number)

        self._prepare_matrix_probe_cards(path)
        misc.compile(cwd=pjoin(path, 'Source'))
        # the P directory makefile includes the analysis options a launch
        # writes: the dummy analysis, as the aMC@NLO launch writes it
        analyse_opts = pjoin(path, 'SubProcesses', 'analyse_opts')
        if not os.path.isfile(analyse_opts):
            with open(analyse_opts, 'w') as stream:
                stream.write('FO_ANALYSE=analysis_dummy.o dbook.o '
                             'open_output_files_dummy.o HwU_dummy.o\n')
        objects = [os.path.basename(source)[:-2] + '.o' for source in
                   sorted(glob.glob(pjoin(subproc, 'matrix_*.f')))]
        objects += ['real_me_chooser.o', 'splitorders_stuff.o',
                    'check_real_cf.o']
        native_objects, native_flags = self._native_probe_links(path, subproc)
        objects += native_objects
        with open(pjoin(subproc, 'check_real_cf.f'), 'w') as stream:
            stream.write(self._real_cf_driver_source())
        misc.compile(objects, cwd=subproc)
        executable = pjoin(subproc, 'check_real_cf')
        subprocess.check_call(
            ['gfortran', '-o', executable] + objects + [
                '-L%s' % pjoin(path, 'lib'), '-ldhelas', '-lmodel'] + native_flags,
            cwd=subproc)
        stdin = '%d\n' % rows[0] + ''.join(
            ' '.join('%.17e' % component for component in momentum) + '\n'
            for momentum in self.GGGGGG_POINT)
        output = subprocess.check_output(
            [executable], input=stdin.encode(), cwd=subproc,
            stderr=subprocess.STDOUT).decode(errors='replace')

        values = {}
        for line in output.splitlines():
            tokens = line.split()
            if len(tokens) == 2 and tokens[0] in ('REAL_SCALAR',
                                                  'REAL_BATCHED'):
                values[tokens[0]] = self._fortran_float(tokens[1])
        self.assertEqual(sorted(values), ['REAL_BATCHED', 'REAL_SCALAR'],
                         output)
        for entry_point, value in sorted(values.items()):
            self.assertAlmostEqual(
                value, self.GGGGGG_MG5AMC_380,
                delta=abs(self.GGGGGG_MG5AMC_380) * 1e-12,
                msg='%s: g g > g g g g at the reference point' % entry_point)

    @staticmethod
    def _mixed_wj_driver_source():
        """Fixed-point real/extra-counterterm oracle for mixed W+j output."""
        return r"""      PROGRAM CHECK_MIXED_WJ
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      INCLUDE 'nFKSconfigs.inc'
      INTEGER I,J,K,NFKSPROCESS
      DOUBLE PRECISION PR(0:3,NEXTERNAL),PB(0:3,NEXTERNAL-1)
      DOUBLE PRECISION WGT,E,E3,E4,MW,S
      DOUBLE PRECISION RET_AMP_SPLIT(AMP_SPLIT_SIZE)
      DOUBLE COMPLEX CNTS(2,NSPLITORDERS)
      LOGICAL TARGET
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      INCLUDE 'fks_info.inc'
      CALL SETPARA('param_card.dat')
      MW=80.419D0
      S=1000D0
      E=(S-MW)/2D0
      PR(:,:)=0D0
      PR(0,1)=S/2D0
      PR(3,1)=S/2D0
      PR(0,2)=S/2D0
      PR(3,2)=-S/2D0
      PR(0,3)=MW
      PR(0,4)=E
      PR(1,4)=E
      PR(0,5)=E
      PR(1,5)=-E
      E3=(S*S+MW*MW)/(2D0*S)
      E4=(S*S-MW*MW)/(2D0*S)
      PB(:,:)=0D0
      PB(0,1)=S/2D0
      PB(3,1)=S/2D0
      PB(0,2)=S/2D0
      PB(3,2)=-S/2D0
      PB(0,3)=E3
      PB(1,3)=E4
      PB(0,4)=E4
      PB(1,4)=-E4
      DO K=1,3
        DO I=1,FKS_CONFIGS
          TARGET=(PDG_TYPE_D(I,1).EQ.1.AND.
     $      PDG_TYPE_D(I,2).EQ.2.AND.PDG_TYPE_D(I,3).EQ.24.AND.
     $      PDG_TYPE_D(I,4).EQ.1.AND.PDG_TYPE_D(I,5).EQ.1).OR.
     $      (PDG_TYPE_D(I,1).EQ.2.AND.
     $      PDG_TYPE_D(I,2).EQ.2.AND.PDG_TYPE_D(I,3).EQ.24.AND.
     $      PDG_TYPE_D(I,4).EQ.1.AND.PDG_TYPE_D(I,5).EQ.2)
          IF (TARGET) THEN
            NFKSPROCESS=I
            WRITE(*,*) 'ORACLE_BEGIN',K,I
            WRITE(*,*) 'ORACLE_PDGS',
     $        (PDG_TYPE_D(I,J),J=1,NEXTERNAL)
            WRITE(*,*) 'ORACLE_INDICES',REAL_FLAVOR_INDEX_D(I),
     $        BORN_FLAVOR_INDEX_D(I),EXTRA_CNT_FLAVOR_INDEX_D(I),
     $        EXTRA_CNT_D(I)
            WRITE(*,*) 'ORACLE_EXTRA_PDGS',
     $        (EXTRA_CNT_PDG_D(I,J),J=1,NEXTERNAL-1)
            WRITE(*,*) 'ORACLE_EXTRA_COLORS',
     $        (EXTRA_CNT_COLOR_D(I,J),J=1,NEXTERNAL-1)
            WRITE(*,*) 'ORACLE_EXTRA_CHARGES',
     $        (EXTRA_CNT_CHARGE_D(I,J),J=1,NEXTERNAL-1)
            CALL SMATRIX_REAL(PR,RET_AMP_SPLIT,WGT)
            WRITE(*,*) 'ORACLE_REAL',WGT
            CALL EXTRA_CNT(PB,EXTRA_CNT_D(I),CNTS)
            DO J=1,NSPLITORDERS
              WRITE(*,*) 'ORACLE_CNT',J,DBLE(CNTS(1,J)),
     $          DIMAG(CNTS(1,J)),DBLE(CNTS(2,J)),
     $          DIMAG(CNTS(2,J))
            ENDDO
            WRITE(*,*) 'ORACLE_END'
          ENDIF
        ENDDO
      ENDDO
      END
"""

    def _run_mixed_wj_driver(self, output_path, subproc_pattern):
        """Compile and run the mixed W+j oracle in one subprocess."""
        found = glob.glob(pjoin(output_path, 'SubProcesses', subproc_pattern))
        self.assertEqual(len(found), 1, found)
        subproc = found[0]
        misc.compile(cwd=pjoin(output_path, 'Source'))

        matrix_objects = [
            os.path.basename(path)[:-2] + '.o' for path in
            sorted(glob.glob(pjoin(subproc, 'matrix_*.f')))]
        counterterm_objects = [
            os.path.basename(path)[:-2] + '.o' for path in
            sorted(glob.glob(pjoin(subproc, 'born_cnt_*.f')))]
        objects = (matrix_objects + counterterm_objects +
                   ['real_me_chooser.o', 'extra_cnt_wrapper.o',
                    'splitorders_stuff.o'])
        native_objects, native_flags = self._native_probe_links(output_path, subproc)
        objects += native_objects
        self.assertTrue(matrix_objects)
        self.assertTrue(counterterm_objects)

        source = pjoin(subproc, 'check_mixed_wj.f')
        from tests.acceptance_tests.test_nlo_real_offload_contract import TestNLORealFortranOracle
        oracle = TestNLORealFortranOracle()
        physical_rows, unused_counts = oracle._real_me_contract(subproc)
        self.assertEqual(sorted(physical_rows),
                         list(range(1, max(physical_rows) + 1)))
        # The runtime table also includes Born/history aliases owned by other
        # native contexts. This probe covers every physical real-provider row.
        driver = self._mixed_wj_driver_source().replace(
            'DO I=1,FKS_CONFIGS', 'DO I=1,%d' % max(physical_rows))
        with open(source, 'w') as fsock:
            fsock.write(driver)
        misc.compile(objects + ['check_mixed_wj.o'], cwd=subproc)

        executable = pjoin(subproc, 'check_mixed_wj')
        libdir = pjoin(output_path, 'lib')
        subprocess.check_call(
            ['gfortran', '-o', executable, 'check_mixed_wj.o'] + objects +
            ['-L%s' % libdir, '-ldhelas', '-lmodel'] + native_flags, cwd=subproc)
        output = subprocess.check_output(
            [executable], cwd=subproc, stderr=subprocess.STDOUT)
        return self._parse_mixed_wj_driver(output.decode(errors='replace'))

    @classmethod
    def _parse_mixed_wj_driver(cls, output):
        """Parse ORACLE_* records, ignoring diagnostics from matrix calls."""
        records = {}
        current = None
        for line in output.splitlines():
            tokens = line.split()
            if not tokens or not tokens[0].startswith('ORACLE_'):
                continue
            tag = tokens[0][7:].lower()
            if tag == 'begin':
                current = {'repeat': int(tokens[1]), 'row': int(tokens[2]),
                           'counterterms': {}}
                continue
            if current is None:
                continue
            if tag in ('pdgs', 'extra_pdgs', 'extra_colors', 'indices'):
                current[tag] = tuple(int(value) for value in tokens[1:])
            elif tag == 'extra_charges':
                current[tag] = tuple(cls._fortran_float(value)
                                     for value in tokens[1:])
            elif tag == 'real':
                current['real'] = cls._fortran_float(tokens[1])
            elif tag == 'cnt':
                current['counterterms'][int(tokens[1])] = tuple(
                    cls._fortran_float(value) for value in tokens[2:])
            elif tag == 'end':
                records.setdefault(current['pdgs'], []).append(current)
                current = None
        return records

    def test_fks_standalone_wj_all_ordered_real_sectors(self):
        """All 152 physical W+j sectors match ungrouped totals/components.

        Six channels used to disappear before export. Exercise their real
        amplitudes as well as the identical-final-state normalization controls
        on sixteen deterministic massive-W phase-space points.
        """
        import random
        import math
        from tests.acceptance_tests import test_nlo_real_offload_contract as offload
        from madgraph.various.rambo import RAMBO, FortranList

        old_random_state = random.getstate()
        random.seed(20261105)
        masses = FortranList(3)
        masses[1], masses[2], masses[3] = 80.419, 0., 0.
        points = []
        try:
            for unused in range(16):
                final, unused_weight = RAMBO(3, 1000., masses)
                points.append([[500., 0., 0., 500.],
                               [500., 0., 0., -500.]] + [
                    [final[4, i], final[1, i], final[2, i], final[3, i]]
                    for i in range(1, 4)])
        finally:
            random.setstate(old_random_state)
        driver = """      PROGRAM CHECK_ORDERED_WJ
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      INCLUDE 'nFKSconfigs.inc'
      INTEGER K,J,M,N,NFKSPROCESS
      DOUBLE PRECISION P(0:3,NEXTERNAL),WGT
      DOUBLE PRECISION RET(AMP_SPLIT_SIZE)
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      INCLUDE 'fks_info.inc'
      CALL SETPARA('param_card.dat')
      READ(*,*) N
      DO K=1,N
        READ(*,*) NFKSPROCESS
        DO J=1,NEXTERNAL
          READ(*,*) (P(M,J),M=0,3)
        ENDDO
        CALL PHYSICAL_REAL(P,RET,WGT)
        WRITE(*,*) 'VALUE',K,FKS_I_D(NFKSPROCESS),
     $    FKS_J_D(NFKSPROCESS),WGT,RET
      ENDDO
      END
"""
        comparisons = {}
        for grouped in (False, True):
            path = pjoin(self.tmpdir, 'wj_ordered_%s' % grouped)
            cmd = self._new_cmd()
            cmd._experimental_nlo_flavor_grouping = grouped
            for line in [
                    'set apply_flavor_grouping %s --no_save' % grouped,
                    'set low_mem_multicore_nlo_generation False --no_save',
                    'import model loop_sm-no_b_mass',
                    'define p = g u c d s b u~ c~ d~ s~ b~',
                    'define j = p', 'generate p p > w+ j [real=QCD]',
                    'output standalone_fortran --fks --limits %s' % path]:
                self._run(cmd, line)
            misc.compile(cwd=pjoin(path, 'Source'))
            registry = json.loads((Path(path) /
                'Source/BornSupport/registry.json').read_text())
            contexts = {context['directory']: context
                        for context in registry['contexts']}
            group_positions = {signed: position
                for members in (cmd._curr_model.get('merged_particles') or {}).values()
                for position, pdg in enumerate(members, 1)
                for signed in (int(pdg), -int(pdg))}
            records = {}
            for sub in sorted(glob.glob(pjoin(path, 'SubProcesses', 'P*'))):
                # Native finalization adds Born/history aliases and rewrites
                # runtime rows. leshouche_info.dat retains the earlier export
                # numbering, so enumerate the registered physical sectors and
                # call the retained provider before native alias routing.
                context = contexts[os.path.basename(sub)]
                oracle = offload.TestNLORealFortranOracle()
                row_to_me, unused_counts = oracle._real_me_contract(sub)
                self.assertEqual(sorted(row_to_me),
                    [sector['sector'] for sector in context['sectors']])
                flat = ' '.join(line[6:].strip() for line in
                    (Path(sub) / 'fks_info.inc').read_text().splitlines()
                    if line and line[0].upper() != 'C')
                def metadata(name, row):
                    match = re.search(r'DATA\s+' + name +
                        r'\s*\(\s*' + str(row) + r'\s*\)\s*/\s*(-?\d+)\s*/',
                        flat, re.I)
                    self.assertIsNotNone(match, (name, row))
                    return int(match.group(1))
                rows = []
                for sector in context['sectors']:
                    row = sector['sector']
                    self.assertEqual(row_to_me[row], sector['real'])
                    indices = (sector['fks']['i'], sector['fks']['j'])
                    self.assertEqual((metadata('FKS_I_D', row),
                        metadata('FKS_J_D', row)), indices)
                    matrix = (Path(sub) /
                        ('matrix_%d.f' % sector['real'])).read_text()
                    bound = re.search(r'NMASK_FLAV\s*=\s*(\d+)', matrix, re.I)
                    self.assertIsNotNone(bound)
                    matrix_flat = ' '.join(line[6:].strip()
                        for line in matrix.splitlines()
                        if line and line[0].upper() != 'C')
                    table = re.search(r'DATA\s+FLAV_TABLE\s*/([^/]+)/',
                                      matrix_flat, re.I)
                    self.assertIsNotNone(table)
                    masks = tuple(map(int, re.findall(r'-?\d+', table.group(1))))
                    self.assertEqual(len(masks), 5 * int(bound.group(1)))
                    flavour = metadata('REAL_FLAVOR_INDEX_D', row)
                    self.assertGreaterEqual(flavour, 1)
                    self.assertLessEqual(flavour, int(bound.group(1)))
                    # Multiple processes in a non-grouped sector are genuine
                    # generation-equivalent channels sharing its sole mask.
                    if len(sector['processes']) > 1:
                        self.assertEqual(int(bound.group(1)), 1)
                    for process in sector['processes']:
                        pdgs = tuple(leg[0] for leg in process[1])
                        self.assertEqual(masks[5*(flavour-1):5*flavour],
                            tuple(group_positions.get(pdg, 1) for pdg in pdgs),
                            (context['directory'], row, pdgs, flavour))
                        rows.append((row, pdgs, indices))
                inputs = []
                binding = []
                for row, pdgs, indices in rows:
                    for point_index, momenta in enumerate(points):
                        binding.append((point_index, pdgs, indices))
                        inputs.append(str(row))
                        inputs.extend(' '.join('%.17e' % x for x in p)
                                      for p in momenta)
                inputs = str(len(binding)) + '\n' + '\n'.join(inputs) + '\n'
                with open(pjoin(sub, 'check_ordered_wj.f'), 'w') as stream:
                    stream.write(driver.replace('PHYSICAL_REAL',
                        oracle._physical_scalar_name(sub)))
                objects = [os.path.basename(p)[:-2] + '.o' for p in
                           sorted(glob.glob(pjoin(sub, 'matrix_*.f')))]
                objects += ['real_me_chooser.o', 'splitorders_stuff.o',
                            'check_ordered_wj.o']
                native_objects, native_flags = self._native_probe_links(path, sub)
                objects += native_objects
                misc.compile(objects, cwd=sub)
                executable = pjoin(sub, 'check_ordered_wj')
                subprocess.check_call(['gfortran', '-o', executable] + objects +
                    ['-L' + pjoin(path, 'lib'), '-ldhelas', '-lmodel'] + native_flags, cwd=sub)
                output = subprocess.check_output([executable], input=inputs,
                    cwd=sub, text=True, stderr=subprocess.STDOUT)
                for line in output.splitlines():
                    words = line.split()
                    if not words or words[0] != 'VALUE':
                        continue
                    point_index, pdgs, indices = binding[int(words[1]) - 1]
                    self.assertEqual(tuple(map(int, words[2:4])), indices)
                    key = (point_index, pdgs, int(words[2]), int(words[3]))
                    self.assertNotIn(key, records)
                    records[key] = tuple(self._fortran_float(x)
                                         for x in words[4:])
            self.assertEqual(len(records), 152 * 16)
            comparisons[grouped] = records
        self.assertEqual(set(comparisons[True]), set(comparisons[False]))
        for key, expected in comparisons[False].items():
            actual = comparisons[True][key]
            self.assertEqual(len(actual), len(expected))
            scale = max(map(abs, expected[1:]))
            for got, reference in zip(actual, expected):
                self.assertTrue(math.isfinite(got) and math.isfinite(reference))
                self.assertAlmostEqual(got, reference,
                    delta=max(abs(reference) * 1e-12, scale * 1e-12, 1e-30),
                    msg='point/PDGs/i/j %s' % (key,))

    def test_fks_standalone_grouped_mixed_wj_tree_values(self):
        """Grouped mixed-QCD/QED W+j real and extra-counterterm rows match
        fixed-flavour output, including physical metadata and repeated calls."""
        model = pjoin(MG5DIR, 'tests', 'input_files', 'LoopSMEWTest')
        grouped_path = pjoin(self.tmpdir, 'grouped_mixed_wj')
        reference_path = pjoin(self.tmpdir, 'fixed_mixed_wj')

        cmd = self._new_cmd()
        self._run(cmd, 'set nlo_mixed_expansion True --no_save')
        cmd._experimental_nlo_flavor_grouping = True
        self._run(cmd, 'set apply_flavor_grouping True --no_save')
        self._run(cmd, 'import model %s' % model)
        self._run(cmd, 'generate p p > w+ j QED^2=4 QCD^2=4 '
                       '[real=QCD QED]')
        self._run(cmd, 'output standalone_fortran --fks --limits %s -f' %
                  grouped_path)

        cmd = self._new_cmd()
        self._run(cmd, 'set nlo_mixed_expansion True --no_save')
        self._run(cmd, 'set apply_flavor_grouping False --no_save')
        self._run(cmd, 'import model %s' % model)
        self._run(cmd, 'generate g u > w+ d QED^2=4 QCD^2=4 '
                       '[real=QCD QED]')
        self._run(cmd, 'output standalone_fortran --fks --limits %s -f' %
                  reference_path)

        grouped = self._run_mixed_wj_driver(grouped_path, 'P*_gQ_wpQ')
        reference = self._run_mixed_wj_driver(reference_path, 'P*_gu_wpd')
        target_pdgs = ((1, 2, 24, 1, 1), (2, 2, 24, 1, 2))
        self.assertEqual(set(grouped), set(target_pdgs))
        self.assertEqual(set(reference), set(target_pdgs))

        expected_real = {
            target_pdgs[0]: 2.4497695572173661e-4,
            target_pdgs[1]: 4.9257600649863040e-4,
        }
        for pdgs in target_pdgs:
            self.assertEqual(len(grouped[pdgs]), 3)
            self.assertEqual(len(reference[pdgs]), 3)
            for actual, fixed in zip(grouped[pdgs], reference[pdgs]):
                self.assertEqual(actual['repeat'], fixed['repeat'])
                self.assertEqual(actual['pdgs'], pdgs)
                self.assertEqual(actual['extra_pdgs'], (22, 2, 24, 1))
                self.assertEqual(actual['extra_colors'], (1, 3, 1, 3))
                for charge, expected in zip(
                        actual['extra_charges'], (0., 2./3., 1., -1./3.)):
                    self.assertClose(charge, expected, rel=1e-12)
                self.assertEqual(actual['extra_pdgs'], fixed['extra_pdgs'])
                self.assertEqual(actual['extra_colors'],
                                 fixed['extra_colors'])
                self.assertEqual(actual['extra_charges'],
                                 fixed['extra_charges'])
                self.assertClose(actual['real'], fixed['real'], rel=1e-12)
                self.assertClose(actual['real'], expected_real[pdgs],
                                 rel=1e-12)
                self.assertEqual(set(actual['counterterms']),
                                 set(fixed['counterterms']))
                for order in actual['counterterms']:
                    for value, expected in zip(
                            actual['counterterms'][order],
                            fixed['counterterms'][order]):
                        self.assertClose(value, expected, rel=1e-12)

        self.assertEqual(grouped[target_pdgs[0]][0]['indices'],
                         (1, 1, 1, 1))
        # Ordered real-flavour preservation inserts the previously discarded
        # permutations before this physical row; its amplitude/CT references
        # above are unchanged.
        self.assertEqual(grouped[target_pdgs[1]][0]['indices'],
                         (5, 1, 1, 1))
        self.assertEqual(reference[target_pdgs[0]][0]['indices'],
                         (1, 1, 0, 1))
        self.assertEqual(reference[target_pdgs[1]][0]['indices'],
                         (1, 1, 0, 1))

        for pdgs in target_pdgs:
            first = grouped[pdgs][0]
            for repeated in grouped[pdgs][1:]:
                self.assertEqual(repeated['indices'], first['indices'])
                self.assertClose(repeated['real'], first['real'], rel=1e-14)
                self.assertEqual(set(repeated['counterterms']),
                                 set(first['counterterms']))
                for order in repeated['counterterms']:
                    for value, expected in zip(
                            repeated['counterterms'][order],
                            first['counterterms'][order]):
                        self.assertClose(value, expected, rel=1e-14)

    def _check_vs_amcatnlo(self, spec):
        """the standalone Born building blocks match the ones computed by the
        code of a plain aMC@NLO output of the same process.

        The trimmed standalone retains scalar kernels, while full native
        output installs registry facades and private providers. Their wrapper
        text intentionally differs. Bind the build to its registry provider
        and compare every physical configuration at sixteen fixed points."""
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
        support = Path(path_nlo) / 'Source/BornSupport'
        registry = json.loads((support / 'registry.json').read_text())
        record = next(item for item in registry['contexts']
                      if item['directory'] == pname)
        provider = support / ('p%d' % record['provider'])
        kernel = provider / 'matrix.f'
        self.assertTrue(kernel.is_file())
        self.assertIn('p%d/matrix.o' % record['provider'],
                      (support / 'providers.mk').read_text().splitlines()[0])
        kernel_hash = hashlib.sha256(kernel.read_bytes()).hexdigest()
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
        self.assertEqual(hashlib.sha256(kernel.read_bytes()).hexdigest(), kernel_hash)
        self.assertGreaterEqual((provider / 'matrix.o').stat().st_mtime_ns,
                                kernel.stat().st_mtime_ns)
        # Keep reproducible source/object/library identities with the probe.
        (support / 'comparison_provenance.json').write_text(json.dumps({
            'context': record['context'], 'provider': record['provider'],
            'source_directory': record['source'], 'source_sha256': kernel_hash,
            'object_sha256': hashlib.sha256((provider / 'matrix.o').read_bytes()).hexdigest(),
            'library_sha256': hashlib.sha256(
                (Path(path_nlo) / ('lib/libmc_born_support' +
                 ('.dylib' if sys.platform == 'darwin' else '.so'))).read_bytes()).hexdigest(),
        }, indent=2) + '\n')
        val_nlo = self.parse_check_fks(self._run_check_fks(born_dir_nlo))

        # 4) identical Born building blocks (same code, same seed)
        self.assertClose(val_sa[0], val_nlo[0], rel=1e-9, msg='BORN')
        self.assertClose(val_sa[1], val_nlo[1], rel=1e-9, msg='BORNTILDE')
        self.assertEqual(set(val_sa[2]), set(val_nlo[2]),
                         'color link sets differ between outputs')
        for key in val_sa[2]:
            self.assertClose(val_sa[2][key], val_nlo[2][key], rel=1e-9,
                             msg='B_ij%s' % (key,))
        points_sa = self.parse_check_fks_points(self._run_check_fks(
            born_dir_sa, points=16, calls=3))
        points_nlo = self.parse_check_fks_points(self._run_check_fks(
            born_dir_nlo, points=16, calls=3))
        self.assertEqual(len(points_sa), 16)
        self.assertPointsClose(points_nlo, points_sa, rel=1e-9)
        for native_point, scalar_point in zip(points_nlo, points_sa):
            native_rows = native_point['configurations']
            scalar_rows = scalar_point['configurations']
            self.assertEqual(len(native_rows), len(scalar_rows))
            for actual, expected in zip(native_rows, scalar_rows):
                for key in ('index', 'pdgs', 'flavor_index'):
                    self.assertEqual(actual[key], expected[key])
                for key in ('born', 'borntilde'):
                    self.assertClose(actual[key], expected[key], rel=1e-9, msg=key)
                self.assertEqual(set(actual['bij']), set(expected['bij']))
                for key in actual['bij']:
                    self.assertClose(actual['bij'][key], expected['bij'][key],
                                     rel=1e-9, msg='B_ij%s' % (key,))

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
                ({'flavor': -1}, '--flavor'),
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
      REAL*8 XP(0:3,5,15),LXP(0:3,5)
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
      XSEC=1D0
      CALL CHECKRES(XSEC,XLIM,WGT,WGTL,XP,LXP,
     &  0,15,1,4,1,2,IRET1)
      CALL CHECKRES2(XSEC,XLIM2,WGT,WGTL2,XP2,LXP2,
     &  0,15,1,1,2,IRET2)
      IF (IRET1.NE.0 .OR. IRET2.NE.0) STOP 3
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
        result = subprocess.run(
            ['./check_nonconvergence'], cwd=test_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('NONCONVERGENCE SAFELY FAILED', result.stdout)

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
        '--limits' standalone output: every grouped partonic family (gg,
        Q Q~, Q~ Q) must pass test_soft_col_limits, the initial-state
        collinear ones included (hadronic beams with the built-in PDF set)."""
        path = pjoin(self.tmpdir, 'pp_ttx_limits')
        self._output_fks_sa('p p > t t~ [QCD]', 'loop_sm', path, limits=True)
        born_dirs = self._assert_limits_passed(path)
        names = [os.path.basename(d) for d in born_dirs]
        self.assertIn('P0_gg_ttx', names)
        self.assertIn('P0_QQx_ttx', names)
        self.assertIn('P0_QxQ_ttx', names)

    def test_fks_standalone_limits_pp_wpj_mixed(self):
        """soft and collinear limits of the mixed QCD/QED expansion of
        p p > w+ j [real=QCD QED] (grouped): every split order of every born
        dir must pass. This covers the charge-linked soft-photon counterterm
        and the initial-state q -> q + g/gamma limits whose photon-Born
        counterterm is the extra counterterm (born_cnt_*.f)."""
        model = pjoin(MG5DIR, 'tests', 'input_files', 'LoopSMEWTest')
        path = pjoin(self.tmpdir, 'pp_wpj_mixed_limits')
        cmd = self._new_cmd()
        self._run(cmd, 'set nlo_mixed_expansion True --no_save')
        cmd._experimental_nlo_flavor_grouping = True
        self._run(cmd, 'set apply_flavor_grouping True --no_save')
        self._run(cmd, 'import model %s' % model)
        self._run(cmd, 'generate p p > w+ j QED^2=4 QCD^2=4 '
                       '[real=QCD QED]')
        self._run(cmd, 'output standalone_fortran --fks --limits %s -f' %
                  path)
        self._run(cmd, 'launch %s -f' % path)
        born_dirs = self._assert_limits_passed(path)
        names = [os.path.basename(d) for d in born_dirs]
        for name in ('P0_gQ_wpQ', 'P0_aQ_wpQ', 'P0_QQx_wpg', 'P0_QQx_wpa'):
            self.assertIn(name, names)


    def test_fks_born_gscale_mixed_orders(self):
        """Native cached Born amplitudes retain every mixed QCD/QED order.

        Compare rescaling with fresh HELAS evaluations in every sector,
        including a sector switch, a preceding single-helicity query and a
        QED change that must reject strong-coupling-only reuse. Counters in
        the temporary private provider distinguish rescaling from fallback.
        """
        path = pjoin(self.tmpdir, 'uux_ttx_mixed')
        # --limits: full run setup (built-in PDF, dummy analysis), the
        # driver links the same objects as test_soft_col_limits
        self._output_fks_sa('u u~ > t t~ QCD^2<=4 QED^2<=4 [real=QCD]',
                            'loop_sm', path, limits=True)
        born_dir = pjoin(path, 'SubProcesses', 'P0_uux_ttx')
        support = Path(path)/'Source/BornSupport'
        registry = json.loads((support/'registry.json').read_text())
        record = next(r for r in registry['contexts']
                      if r['directory'] == os.path.basename(born_dir))
        # The exported Born has two amplitude orders and all three squared
        # orders. The native provider certifies the actual diagram powers.
        self.assertEqual((record['namps'], record['nsqamps']), (2, 3))
        qcd = record['orders'].index('QCD')
        self.assertEqual([o[qcd] for o in record['amp_orders'][:record['nsqamps']]],
                         [4, 2, 0])
        self.assertEqual(record['helicity_matrices'][0]['amplitude_degrees'],
                         [2, 0, 0])
        signature = json.loads((support/'helicity_signature.json').read_text())
        self.assertIn(signature['mode'], ('homogeneous_G', 'exact_compact_state'))
        certified_scaling = signature['mode'] == 'homogeneous_G'
        self.assertEqual(signature['signature'] is not None, certified_scaling)
        # The QED input belongs to the cold configuration. Changing it must
        # select a new configuration before any strong-coupling-only reuse.
        qed_coupling = 'GC_2'
        layout = json.loads((support/'model_state_layout.json').read_text())
        self.assertIn(qed_coupling.lower(),
                      [name for name, _, _ in layout['fixed_members']])
        self.assertNotIn(qed_coupling.lower(),
                         [name for name, _, _ in layout['dynamic_members']])
        nlinks = len(record['links'])

        # Instrument only this generated DSO. CALCULATEDBORN belongs to the
        # legacy API and cannot invalidate the native point cache. A one-shot
        # reset supplies genuinely fresh references using the same matrix.
        types = support/'mc_born_types.f90'
        text, count = re.subn(r'(?m)^  implicit none$',
            '  implicit none\n  logical,save :: born_test_fresh=.false.\n'
            '  integer,save :: born_test_scaled=0,born_test_helas=0',
            types.read_text())
        self.assertEqual(count, 1)
        types.write_text(text)
        provider = support/('p%d' % record['provider'])
        adapter = []
        resets = scales = 0
        for line in born_support.statements((provider/'adapter.f').read_text()):
            if line.lower().startswith('model_changed='):
                # The lane-sized cache is allocated on the first request,
                # below MODEL_CHANGED. A fresh first call has no entries to
                # invalidate; later calls still clear every cached point.
                adapter += ['if(born_test_fresh)then',
                            'if(allocated(points))then',
                            'points%valid=.false.', 'points%age=0', 'endif',
                            'active_slot=0',
                            'born_test_fresh=.false.', 'endif']
                resets += 1
            if line.lower() == 'saveamp(:,i)=saveamp(:,i)*factors':
                adapter.append('if(i.eq.1)born_test_scaled=born_test_scaled+1')
                scales += 1
            adapter.append(line)
        self.assertEqual((resets, scales), (1, 1))
        born_support.write_fortran(provider/'adapter.f', adapter)
        matrix = []
        helas_calls = 0
        for line in born_support.statements((provider/'matrix.f').read_text()):
            if re.match(r'(?i)\s*call\s+(?!mcb\d+_)\w+\s*\(', line):
                matrix.append('born_test_helas=born_test_helas+1')
                helas_calls += 1
            matrix.append(line)
            if re.match(r'(?i)\s*(?:[\w*]+\s+)*?'
                        r'(?:subroutine|function)\s+\w+', line):
                matrix.append('use mc_born_types,only:born_test_helas')
        self.assertGreater(helas_calls, 0)
        born_support.write_fortran(provider/'matrix.f', matrix)
        misc.compile(cwd=str(support))

        driver = (Path(__file__).resolve().parents[1]/'input_files'/
                  'fks_born_gscale_driver.f').read_text()
        driver = driver.replace('QED_COUPLING', qed_coupling).replace(
            'NLINKS_VALUE', str(nlinks)).replace(
            'LINK_M_VALUES', ','.join(str(m) for m, n in record['links'])).replace(
            'LINK_N_VALUES', ','.join(str(n) for m, n in record['links']))
        # Snapshot reconstruction retains the explicit-buffer API; the native
        # scalar facade instead queries the currently active model state.
        driver = driver.replace('call sborn_hel(', 'call sborn_hel_buffered(')
        driver = driver.replace('call sborn_onehel(pb,nhel,1,w,ret_saveamp)',
                                'call sborn_onehel(pb,nhel,1,w)')
        # The explicit-array exporter keeps one source per link. Adapt only
        # this test driver's scalar convenience API, preserving every link
        # comparison and all upstream rescaling/refusal assertions.
        driver += '''
      subroutine sb_sf_link(p,ilink,w)
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'born_nhel.inc'
      include 'orders.inc'
      double precision p(0:3,nexternal-1),w
      integer ilink
      double complex saveamp(ngraphs,max_bhel)
      common /to_saveamp/ saveamp
      double complex ret(amp_split_size,2,nsplitorders)
'''
        for ilink in range(1, nlinks + 1):
            driver += ('      %s (ilink.eq.%d) then\n'
                       '        call sb_sf_%03d(p,w,ret,saveamp)\n' %
                       ('if' if ilink == 1 else 'elseif', ilink, ilink))
        driver += '      else\n      stop 97\n      endif\n      end\n'
        open(pjoin(born_dir, 'test_born_gscale.f'), 'w').write(driver)
        makefile = open(pjoin(born_dir, 'makefile')).read()
        makefile += (
            '\nGSCALETEST= $(filter-out test_soft_col_limits.o,$(TEST)) '
            'test_born_gscale.o\n'
            'test_born_gscale: $(GSCALETEST) mint_module.o\n'
            '\t$(FC) $(LDFLAGS) -o test_born_gscale $(GSCALETEST) '
            '$(APPLLIBS) $(LINKLIBS) $(FO_EXTRAPATHS) $(FO_EXTRALIBS) '
            '$(FJLIBS)\n'
            'test_born_gscale.o: mint_module.o genps_fks.o\n')
        (Path(born_dir)/'makefile_gscale').write_text(makefile)
        misc.compile(['-f', 'makefile_gscale', 'test_born_gscale'],
                     cwd=born_dir)
        checked = subprocess.run(
            [pjoin(born_dir, 'test_born_gscale')], cwd=born_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = checked.stdout.decode(errors='replace')
        self.assertEqual(checked.returncode, 0, out)

        npoints = int(re.search(r'GSCALE NPOINTS\s+(\d+)', out).group(1))
        self.assertGreater(npoints, 1, out)
        self.assertEqual(npoints, 2*len(record['sectors']), out)
        nvalues = int(re.search(r'GSCALE NVALUES\s+(\d+)', out).group(1))
        norders = len(record['amp_orders'])
        self.assertEqual(nvalues, 1+norders+4*len(record['orders'])*(1+norders)
                         +record['ngraphs']+record['ncolor']
                         +len(record['links'])*(1+norders), out)
        modes = {}
        for mode, worst, count, helas in re.findall(
                r'GSCALE MODE\s+(\d)\s+(\S+)\s+(\d+)\s+(\d+)', out):
            modes[int(mode)] = (float(worst), int(count), int(helas))
        self.assertEqual(sorted(modes), [0, 1, 2, 3], out)
        for mode in (0, 1, 2):
            worst, count, helas = modes[mode]
            if certified_scaling:
                self.assertEqual(count, npoints, 'mode %d: %s' % (mode, out))
                self.assertEqual(helas, 0, 'mode %d recomputed HELAS: %s' % (mode, out))
            else:
                # Exact model-state certification deliberately selects fresh
                # HELAS rather than applying an uncertified strong rescaling.
                self.assertEqual(count, 0, 'mode %d: %s' % (mode, out))
                self.assertGreater(helas, 0, 'mode %d: %s' % (mode, out))
            # Both paths must agree with independently fresh evaluations.
            self.assertLess(worst, 1e-12, 'mode %d: %s' % (mode, out))
        worst, count, helas = modes[3]
        # refusal: never rescaled, identical to a fresh evaluation with
        # the changed coupling, which does change the Born
        self.assertEqual(count, 0, out)
        self.assertGreater(helas, 0, out)
        self.assertEqual(worst, 0., out)
        shift = float(re.search(r'GSCALE REFUSAL_SHIFT\s+(\S+)',
                                out).group(1))
        self.assertGreater(shift, 1e-6, out)
        snapshot = float(re.search(r'GSCALE SAVED_SNAPSHOT\s+(\S+)',
                                   out).group(1))
        self.assertLess(snapshot, 1e-12, out)



if __name__ == '__main__':
    unittest.main()
