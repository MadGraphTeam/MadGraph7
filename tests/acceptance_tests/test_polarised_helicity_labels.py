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
"""A polarised process must name its helicity configurations consistently.

A single helicity configuration is always named by its row in
get_helicity_matrix(): madevent stores the row it sampled and decodes the LHE
helicities from the canonical-code tables (export_v4._helstate_data), and the
standalone callers -- the reweighting first -- ask SMATRIXHEL for a row. The
canonical code therefore has to BE the row. Built over the full particle
states instead of the polarisation-restricted lists, it was not as soon as one
leg was polarised: p p > w+{0} j wrote W+ helicity -1 in most of its events,
and the per-helicity rows of p p > w+{T} j j added up to 1% of the summed
matrix element.
"""
from __future__ import absolute_import

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import madgraph
import madgraph.interface.master_interface as cmd_interface
from madgraph import MG5DIR

pjoin = os.path.join

# c u > w+ u s (event 43 of the reweighting report) and its three other leg
# orderings: slot j holds leg PERMS[name][j]. The flavour-grouped table holds
# one of the four, the others go through all_matrix.f's flavour order repair.
PDGS = [4, 2, 24, 2, 3]
PERMS = {'id': [0, 1, 2, 3, 4], 'ini': [1, 0, 2, 3, 4],
         'fin': [0, 1, 2, 4, 3], 'both': [1, 0, 2, 4, 3]}
# SMATRIXHEL undoes the initial-state helicity average: 2 x 2 quark states
HELAVGFACTOR = 4

# Per-helicity probe of the combined f2py module. Run in its own interpreter:
# every output builds a module called all_matrix2py, and two of them in one
# process would share their Fortran common blocks.
_ROW_PROBE = r'''
import json, os, random, sys
import numpy as np
sys.path.insert(0, %(mg5dir)r)
os.chdir(%(subproc)r)
sys.path.insert(0, %(subproc)r)
from madgraph.various import rambo
import all_matrix2py as m

m.initialise(os.path.abspath('../Cards/param_card.dat'))
# the q q > w+ q q matrix element and its helicity table, rows in order
pdg_order = [list(row) for row in m.get_pdg_order()[0]]
slot = pdg_order.index([81, 81, 24, 81, 81])
prefix = m.get_prefix()[slot].decode().strip().lower()
nhel = np.array(getattr(m, 'py_%%sget_nhel_idx' %% prefix)(1)[1])
# the leg orderings its flavour table holds; the others go through the repair
nflav = getattr(m, 'py_%%sget_flavor_layout' %% prefix)()[0]
tabulated = set(tuple(int(pdg) for pdg in
                      getattr(m, 'py_%%sget_pdg_for_flavor' %% prefix)(flav))
                for flav in range(1, nflav + 1))

# One fixed point. The W+ only needs to be near its mass shell: everything
# below compares the module with itself.
random.seed(7)
masses = rambo.FortranList(3)
masses[1], masses[2], masses[3] = 80.419, 0., 0.
final, _ = rambo.RAMBO(3, 1000., masses)
p = [[500., 0., 0., 500.], [500., 0., 0., -500.]] + \
    [[final[(4, i)], final[(1, i)], final[(2, i)], final[(3, i)]]
     for i in (1, 2, 3)]

result = {'rows': [[int(h) for h in nhel[:, r]] for r in range(nhel.shape[1])],
          'orderings': {},
          'repaired': sorted(name for name, perm in %(perms)r.items()
                             if tuple(%(pdgs)r[k] for k in perm) not in tabulated)}
for name, perm in %(perms)r.items():
    pdgs = [%(pdgs)r[k] for k in perm]
    mom = np.array([p[k] for k in perm]).T
    result['orderings'][name] = {
        'sum': m.smatrixhel(pdgs, -1, mom, 0.118, 0., -1),
        'rows': [m.smatrixhel(pdgs, -1, mom, 0.118, 0., r)
                 for r in range(1, nhel.shape[1] + 1)]}
print('ROWPROBE ' + json.dumps(result))
'''


class TestPolarisedHelicityLabels(unittest.TestCase):
    """Polarised helicity labels: standalone rows and madevent LHE helicities.

    Each test generates and compiles full outputs, so these are slow tests.
    """

    debugging = getattr(unittest, 'debug', False)

    def setUp(self):
        self.cmd = cmd_interface.MasterCmd()
        self.cmd.no_notification()
        self.tmpdir = tempfile.mkdtemp(prefix='pol_hel_')

    def tearDown(self):
        if not self.debugging and os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def call(self, command, cwd=None):
        if self.debugging:
            return subprocess.call(command, cwd=cwd)
        with open(os.devnull, 'w') as devnull:
            return subprocess.call(command, stdout=devnull, stderr=devnull,
                                   cwd=cwd)

    # ------------------------------------------------------------------
    # standalone: SMATRIXHEL per helicity row
    # ------------------------------------------------------------------
    def probe_standalone(self, name, options=''):
        """Output p p > w+{T} j j as a flavour-grouped standalone with an f2py
        module, and return the per-row probe of its q q > w+ q q matrix
        element. Skips when the f2py build backend is unavailable."""
        outdir = pjoin(self.tmpdir, name)
        self.cmd.exec_cmd('set automatic_html_opening False')
        self.cmd.exec_cmd('set apply_flavor_grouping True')
        self.cmd.exec_cmd('import model sm')
        self.cmd.exec_cmd('generate p p > w+{T} j j')
        self.cmd.exec_cmd('output standalone_fortran %s -f --prefix=int %s'
                          % (outdir, options))
        subproc = pjoin(outdir, 'SubProcesses')
        retcode = self.call(['make', 'all_matrix2py.so'], cwd=subproc)
        if retcode != 0 or not any(f.startswith('all_matrix2py')
                                   and f.endswith('.so')
                                   for f in os.listdir(subproc)):
            raise unittest.SkipTest('Could not build the f2py module in %s '
                                    '(f2py/numpy build backend unavailable)'
                                    % subproc)
        script = pjoin(self.tmpdir, name + '_probe.py')
        with open(script, 'w') as fsock:
            fsock.write(_ROW_PROBE % {'mg5dir': MG5DIR, 'subproc': subproc,
                                      'perms': PERMS, 'pdgs': PDGS})
        output = subprocess.Popen([sys.executable, script],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  cwd=subproc).communicate()[0].decode()
        lines = [l for l in output.splitlines() if l.startswith('ROWPROBE ')]
        self.assertTrue(lines, 'no result from the f2py probe:\n%s' % output)
        return json.loads(lines[-1][len('ROWPROBE '):])

    def assertClose(self, value, reference, msg):
        self.assertLessEqual(abs(value - reference),
                             1e-10 * max(abs(value), abs(reference), 1e-300),
                             '%s: %r vs %r' % (msg, value, reference))

    def tabulated(self, probe):
        """The leg orderings the flavour table holds. A per-helicity row is
        only checked there: an ordering that goes through all_matrix.f's
        flavour order repair needs the requested row re-keyed through the
        leg permutation, which this branch does not do (SMATRIXHEL answers 0
        there for every row, polarised or not)."""
        names = [name for name in PERMS if name not in probe['repaired']]
        self.assertTrue(names, 'no ordering is tabulated')
        return names

    def check_rows(self, probe):
        """The rows of every tabulated ordering are the summed matrix element
        taken apart, and the same configuration in every leg order."""
        rows = [tuple(row) for row in probe['rows']]
        row_of = dict((row, i) for i, row in enumerate(rows))
        names = self.tabulated(probe)
        ref_perm = PERMS[names[0]]
        base = probe['orderings'][names[0]]['rows']
        self.assertTrue(any(base), 'every helicity row is zero')
        for name in names:
            perm = PERMS[name]
            ordering = probe['orderings'][name]
            self.assertClose(sum(ordering['rows']) / HELAVGFACTOR,
                             ordering['sum'],
                             'ordering %s: sum of the rows' % name)
            # entry j of a row in this leg order is the entry, in the
            # reference order, of the reference slot holding leg perm[j]
            slot_of = dict((leg, j) for j, leg in enumerate(ref_perm))
            for r, value in enumerate(base):
                twin = row_of[tuple(rows[r][slot_of[k]] for k in perm)]
                self.assertClose(ordering['rows'][twin], value,
                                 'ordering %s, row %d' % (name, r + 1))

    def test_standalone_polarised_rows(self):
        """SMATRIXHEL(row) on the plain and the helicity-recycled template."""
        plain = self.probe_standalone('plain')
        self.check_rows(plain)

        # The recycled template keeps its own copy of the allowed-code list;
        # its rows must be the plain ones. (Its summed call is not compared:
        # that is a separate matter of the recycled SMATRIXHEL.)
        recycled = self.probe_standalone('recycled', '--hel_recycling=True')
        self.assertEqual(recycled['rows'], plain['rows'])
        for name in self.tabulated(plain):
            for r, (value, reference) in enumerate(zip(
                    recycled['orderings'][name]['rows'],
                    plain['orderings'][name]['rows'])):
                self.assertClose(value, reference, 'recycled, ordering %s, '
                                 'row %d' % (name, r + 1))

    # ------------------------------------------------------------------
    # madevent: the helicities written to the LHE
    # ------------------------------------------------------------------
    def check_madevent_helicities(self, name, options=''):
        """p p > w+{0} j: every event must carry a longitudinal W+, and the
        massless quark line its single chirality (quarks -1, antiquarks +1)."""
        from madgraph.various import lhe_parser
        outdir = pjoin(self.tmpdir, name)
        card = pjoin(self.tmpdir, name + '.txt')
        with open(card, 'w') as fsock:
            fsock.write('import model sm\n'
                        'generate p p > w+{0} j\n'
                        'output madevent %s -f %s\n'
                        'launch\n'
                        'set nevents 100\n'
                        'set iseed 13\n'
                        'set pdlabel nn23lo1\n'
                        'set use_syst False\n'
                        'set systematics_program none\n' % (outdir, options))
        self.call([sys.executable, pjoin(MG5DIR, 'bin', 'madgraph'), card])

        lhe = pjoin(outdir, 'Events', 'run_01', 'unweighted_events.lhe.gz')
        self.assertTrue(os.path.isfile(lhe),
                        'madevent produced no LHE file (%s)' % lhe)
        nevt = 0
        for event in lhe_parser.EventFile(lhe):
            nevt += 1
            helicities = [(part.pid, int(round(part.helicity)))
                          for part in event]
            self.assertEqual([h for pid, h in helicities if pid == 24], [0],
                             'event %d: %s' % (nevt, helicities))
            for pid, hel in helicities:
                if 1 <= abs(pid) <= 6:
                    self.assertEqual(hel, -1 if pid > 0 else 1,
                                     'event %d: %s' % (nevt, helicities))
        self.assertGreater(nevt, 50, 'too few events (%d)' % nevt)

    def test_madevent_polarised_helicities(self):
        """With the default helicity recycling."""
        self.check_madevent_helicities('w0j')

    def test_madevent_polarised_helicities_no_recycling(self):
        """Without helicity recycling: the stored helicity is the plain row."""
        self.check_madevent_helicities('w0j_norecycle', '--hel_recycling=False')
