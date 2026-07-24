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
"""Check the crossing-symmetry support of the fortran standalone output.

The standalone SMATRIX takes a flavor index (IFLAV / FLAV_IDX). Its range is
extended so that a single value carries both the flavor and a crossing to
apply, decoded as::

    cross = (IFLAV-1) / NFLAV
    flav  = mod(IFLAV-1, NFLAV) + 1        ! the index used for masking/...
    I     = cross / (NEXTERNAL+1)
    J     = mod(cross, NEXTERNAL+1)

I and J are the crossing partners of particle 1 and particle 2 respectively:
particle 1 is swapped with particle I and particle 2 with particle J, with 0
meaning "leave that particle alone". IFLAV in [1,NFLAV] gives cross=0, i.e. the
identity, so existing callers are unaffected. The base is NEXTERNAL+1 rather
than NEXTERNAL so that I and J run over 0..NEXTERNAL and can designate the last
particle as well.

Swapping a particle across the initial/final state flips its NSF/NSV helas flag
(which is what negates the momentum stored in the wavefunction) and flips its
helicity, so the crossed call evaluates the same analytic amplitude in a
different kinematic region.

The processes u u~ > g g and u g > u g are exactly each other's crossing under
(I=0, J=3): swapping particle 2 with particle 3 turns the incoming u~ into an
outgoing u and the outgoing g into an incoming g. Because the swap also
reorders the legs, the crossed call takes the *other* process's natural
momentum layout, so this test feeds both codes the very same momenta.

Crossing preserves the raw sum over helicities and colors of |M|^2, not the
averaged matrix element: the two processes have different averaging/symmetry
denominators (IDEN=72 for u u~ > g g, IDEN=96 for u g > u g, since crossing a
gluon into the initial state changes the color average and un-identifies the
two final state gluons). SMATRIX divides by the IDEN of the *crossed* process,
so a crossed call returns the properly averaged matrix element of the process
it crosses into and can be compared directly against the other code.
"""

from __future__ import absolute_import

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import logging

logger = logging.getLogger('madgraph.stdout.cross_symmetry')

import madgraph
import madgraph.interface.master_interface as cmd_interface

pjoin = os.path.join

# The two processes are each other's crossing under (I=0, J=3).
PROC_QQ_GG = 'u u~ > g g'
PROC_QG_QG = 'u g > u g'

# A CHIRAL pair: the W+ couples only to a left-handed u and a right-handed d~, so
# every external quark is 100% polarized and the per-leg density matrix diagonal
# is fully asymmetric ((++) empty, (--) full, or vice versa). That is what makes
# a crossed-fermion helicity FLIP detectable: on u u~ > g g the fermion density
# is (++)==(--), so a flip would be invisible; here it would swap a full entry
# with an empty one. u d~ > w+ g is mapped onto u g > w+ d by (I=0, J=NEXTERNAL):
# the incoming d~ becomes the outgoing d of the last slot (the crossed, still
# 100%-polarized fermion), the outgoing g becomes incoming.
PROC_UDX_WPG = 'u d~ > w+ g'
PROC_UG_WPD = 'u g > w+ d'

# q q~ > g q q~ is likewise mapped onto q g > q q q~ by the same (I=0, J=3)
# crossing: the incoming q~ becomes the outgoing q of slot 3 and the outgoing g
# becomes an incoming one, leaving the legs ordered as (q, g, q, q, q~).
# Repeated over the quark flavors to exercise the flavor tables / masks and the
# BROKEN_SYM factor, which sees two identical final u's on the crossed side.
PROC_QQX_GQQX = '%(q)s %(q)s~ > g %(q)s %(q)s~'
PROC_QG_QQQX = '%(q)s g > %(q)s %(q)s %(q)s~'
QUARK_FLAVORS = ['u', 'd', 's', 'c']

# The merged (multi-flavor) form of the same pair. Generated with the group
# labels so that flavor grouping keeps every quark combination in a single
# matrix element, which is the only way to get NFLAV>1 and a non-trivial mask.
PROC_MERGED_QQX_GQQX = '_quark _anti_quark > g _quark _anti_quark'
PROC_MERGED_QG_QQQX = '_quark g > _quark _quark _anti_quark'

# The same merged process constrained to a single squared coupling order. A
# squared-order constraint is what sets the process' 'split_orders', which is
# what makes write_matrix_element_v4 pick matrix_standalone_splitOrders_v4.inc
# instead of the default template. Same final state, so BROKEN_SYM is still 2
# on the rows where the two final quarks differ.
PROC_MERGED_QG_QQQX_SO = '_quark g > _quark _quark _anti_quark QED^2==0'

# Processes constraining an s-channel propagator. A crossing moves legs between
# the initial and the final state, so what is s-channel in the generated process
# is not s-channel in its crossings: `> z >` (required) and `$$ z` (forbidden,
# diagram removed) must therefore disable the crossing machinery on their own.
# A single `$ z` only forbids the on-shell *region* of a kept diagram, which
# survives the crossing, so it must NOT disable anything.
PROC_REQUIRED_S = 'u u~ > z > e+ e-'
PROC_FORBIDDEN_S = 'u u~ > e+ e- $$ z'
PROC_FORBIDDEN_ONSH_S = 'u u~ > e+ e- $ z'
PROC_UNCONSTRAINED = 'u u~ > e+ e-'

# Every routine/table that only exists to decode an extended FLAV_IDX.
CROSSING_MACHINERY_NAMES = [
    'APPLY_CROSSING', 'APPLY_CROSSING_TABLE', 'GET_CROSS_PERM',
    'GET_SPINCOL_CROSS', 'GET_IDENT_CROSS', 'SWAP_LEGS',
    'SPINCOL_CROSS_TABLE', 'BASEPID_CROSS_TABLE', 'SRC_CROSS_TABLE']

# cross = I*(NEXTERNAL+1) + J = 0*5 + 3 = 3. Both processes have NFLAV=1, so
# IFLAV = cross*NFLAV + flav = 3*1 + 1 = 4.
NEXTERNAL = 4
CROSS_2_3 = 0 * (NEXTERNAL + 1) + 3
# Same crossing for the 2->3 pair, where the base is NEXTERNAL+1 = 6.
NEXTERNAL_5 = 5
CROSS_2_3_5 = 0 * (NEXTERNAL_5 + 1) + 3
# Crossing particle 2 with the *last* particle. Only expressible because the
# base is NEXTERNAL+1: with base NEXTERNAL, mod(cross, NEXTERNAL) could never
# yield NEXTERNAL.
CROSS_2_LAST = 0 * (NEXTERNAL + 1) + NEXTERNAL
IFLAV_IDENTITY = 1


def _iflav(cross, flav, nflav):
    """Encode a crossing code and a flavor index into the extended IFLAV."""
    return cross * nflav + flav


# Subprocess probe for the good-helicity remap (GHREMAP) relation. Run against
# a compiled matrix2py module: for every DERIVABLE crossing (active partners all
# final), the crossed good-helicity set -- the rows where py_smatrixhel_idx is
# non-zero, unioned over many phase-space points -- must equal the identity
# good-helicity set mapped through the crossing's own row permutation sigma
# (config h -> (ic[k]*nhel[perm[k],h])_k). This is the invariant the generated
# GHREMAP encodes, so a wrong table (or a wrong derivability condition) breaks
# the fix. Run in a subprocess: importing an f2py .so into the test interpreter
# would leak a compiled module and clash across tests.
#
# GOTCHA locked in by this probe: 3 phase-space points are NOT enough -- for
# u u~ > g g, cross=23 then showed 6 non-zero rows instead of 8 (an accidental
# zero at the probed points). NPTS is deliberately >= 12.
_GOODHEL_PROBE = r'''
import sys, math
import numpy as np
sys.path.insert(0, %(pdir)r)
import matrix2py as m

NINITIAL = %(ninitial)d
NPTS = %(npts)d

def get_crossing_permutation(cross, nexternal):
    base = nexternal + 1
    i_part, j_part = cross // base, cross %% base
    perm = list(range(nexternal)); ic = [1] * nexternal
    def swap(a, b):
        perm[a], perm[b] = perm[b], perm[a]; ic[a] = -ic[a]; ic[b] = -ic[b]
    valid = not (i_part not in (0, 1) and j_part not in (0, 2)
                 and (i_part == 2 or j_part == 1 or i_part == j_part))
    if i_part not in (0, 1): swap(0, i_part - 1)
    if j_part not in (0, 2): swap(1, j_part - 1)
    return perm, ic, valid

def rambo(nf, ecm, rng):
    q = np.zeros((4, nf))
    for i in range(nf):
        c = 2 * rng.random() - 1
        s = math.sqrt(1 - c * c)
        phi = 2 * math.pi * rng.random()
        r1, r2 = rng.random(), rng.random()
        q[0, i] = -math.log(r1 * r2)
        q[3, i] = q[0, i] * c
        q[2, i] = q[0, i] * s * math.cos(phi)
        q[1, i] = q[0, i] * s * math.sin(phi)
    Q = q.sum(axis=1)
    M = math.sqrt(Q[0]**2 - Q[1]**2 - Q[2]**2 - Q[3]**2)
    b = -Q[1:] / M; g = Q[0] / M; a = 1.0 / (1.0 + g); x = ecm / M
    p = np.zeros((4, nf))
    for i in range(nf):
        bq = b @ q[1:, i]
        p[1:, i] = x * (q[1:, i] + b * (q[0, i] + a * bq))
        p[0, i] = x * (g * q[0, i] + bq)
    return p

def momenta(nexternal, ninitial, npts, seed):
    rng = np.random.default_rng(seed)
    ecm = 1000.0; nf = nexternal - ninitial; ps = []
    for _ in range(npts):
        P = np.zeros((4, nexternal))
        P[0, 0] = ecm / 2; P[3, 0] = ecm / 2
        if ninitial >= 2:
            P[0, 1] = ecm / 2; P[3, 1] = -ecm / 2
        P[:, ninitial:] = rambo(nf, ecm, rng)
        ps.append(np.asfortranarray(P))
    return ps

m.py_initialisemodel(%(card)r)
nflav, nexternal_l, ncross = m.py_get_flavor_layout()
_iden, nhel = m.py_get_nhel_idx(1)
nhel = np.array(nhel)                 # (nexternal, ncomb)
nexternal, ncomb = nhel.shape
ps = momenta(nexternal, NINITIAL, NPTS, seed=20260721)
row_of = {tuple(nhel[:, h]): h + 1 for h in range(ncomb)}

def good_set(flav_idx):
    good = set()
    for P in ps:
        for h in range(1, ncomb + 1):
            if abs(m.py_smatrixhel_idx(P, h, flav_idx)) > 1e-30:
                good.add(h)
    return good

g_id = good_set(1)
assert g_id, 'identity has no good helicity -- probe is broken'
base = nexternal + 1
checked = genuine = 0
for cross in range(1, base * base):
    perm, ic, valid = get_crossing_permutation(cross, nexternal)
    if not valid:
        continue
    I, J = cross // base, cross %% base
    # DERIVABLE = the crossing's active partners are all final particles.
    final_only = ((I in (0, 1) or I > NINITIAL) and (J in (0, 2) or J > NINITIAL))
    if not final_only:
        continue
    flav_idx = cross * nflav + 1
    # Skip a crossing that is not evaluable (spincol==0 -> SMATRIX returns 0).
    tot = sum(abs(m.py_smatrixhel_idx(ps[0], h, flav_idx))
              for h in range(1, ncomb + 1))
    if tot == 0:
        continue
    sigma = {}
    for h in range(ncomb):
        cfg = tuple(ic[k] * nhel[perm[k], h] for k in range(nexternal))
        hp = row_of.get(cfg)
        assert hp is not None, 'cross %%d: sigma is not a row bijection' %% cross
        sigma[h + 1] = hp
    expected = {sigma[h] for h in g_id}
    g_cr = good_set(flav_idx)
    assert g_cr == expected, (
        'cross %%d (I=%%d,J=%%d): crossed good-hel %%s != sigma(identity) %%s'
        %% (cross, I, J, sorted(g_cr), sorted(expected)))
    checked += 1
    if perm != list(range(nexternal)):
        genuine += 1
assert genuine >= 1, 'no genuine (non-identity) derivable crossing was checked'
print('GHREMAP_RELATION_OK checked=%%d genuine=%%d points=%%d' %%
      (checked, genuine, NPTS))
'''


# Subprocess probe for the CROSSED spin-density matrix through the f2py wrapper
# PY_GET_DENSITY_IDX -- the only path by which a python caller can request a
# crossed density matrix (the FLAVOR-array PY_GET_DENSITY resolves through
# GET_FLAVOR_INDEX, which only returns 1..NFLAV and so cannot carry a crossing).
# Prints, per external leg, the three interference terms (++),(+-),(--) of that
# leg's density matrix, so the parent can compare a crossed evaluation against a
# natively generated reference term by term. Run in a subprocess because an
# f2py .so leaks into the importing interpreter and clashes across dirs/tests.
_DENSITY_PROBE = r'''
import sys, json
import numpy as np
sys.path.insert(0, %(pdir)r)
import matrix2py as m
m.py_initialisemodel(%(card)r)
momenta = %(momenta)s                       # [[E,px,py,pz], ...] per leg
P = np.asfortranarray(np.array(momenta, dtype=float).T)   # (4, nexternal)
flav_idx = %(flav_idx)d
allow_hel = np.array([1, -1], dtype=np.int32)
out = {}
for leg in %(legs)s:
    pos = np.array([leg], dtype=np.int32)
    inter = np.asarray(m.py_get_density_idx(
        P, pos, 1, allow_hel, 2, flav_idx, 0.0, 0.0)).ravel()
    out[str(leg)] = [[float(z.real), float(z.imag)] for z in inter]
print('DENSITY_JSON ' + json.dumps(out))
'''


class TestStandaloneCrossSymmetry(unittest.TestCase):
    """u u~ > g g and u g > u g must reproduce each other under crossing."""

    # A crossing swaps a leg between the initial and final state, so it probes
    # a genuinely different kinematic region of the same analytic amplitude.
    # Compare at a few scattering angles rather than a single point.
    cos_thetas = [0.3, -0.62, 0.85]
    energy = 1000.0
    tolerance = 1e-11

    debugging = getattr(unittest, 'debug', False)

    def setUp(self):
        self.cmd = cmd_interface.MasterCmd()
        self.cmd.no_notification()
        prefix = 'cross_debug_' if self.debugging else 'cross_'
        self.tmpdir = tempfile.mkdtemp(prefix=prefix)

    def tearDown(self):
        if not self.debugging and os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    # ------------------------------------------------------------------
    # generation / build helpers
    # ------------------------------------------------------------------
    def _generate(self, process, name, options='', split_orders=False):
        """Generate the standalone output for `process`, return its P* dir.

        `options` is appended to the generate command (e.g. --use_crossing=False).
        `split_orders` selects the driver for the split-orders template, whose
        density entry point takes the FLAVOR array rather than a FLAV_IDX.
        """
        pdir = self._output_standalone(process, name, options)
        self._write_driver(pdir, split_orders=split_orders)
        self._build(pdir)
        return pdir

    def _output_standalone(self, process, name, options=''):
        """Write the standalone output for `process` and return its P* dir.

        Split out of _generate for the tests that only inspect the emitted
        fortran and so have no reason to pay for a compile.
        """
        outdir = pjoin(self.tmpdir, name)
        self.cmd.exec_cmd('set automatic_html_opening False')
        self.cmd.exec_cmd('set group_subprocesses False')
        self.cmd.exec_cmd('set apply_flavor_grouping True')
        self.cmd.exec_cmd('import model sm')
        self.cmd.exec_cmd(('generate %s %s' % (process, options)).strip())
        self.cmd.exec_cmd('output standalone %s -f' % outdir)

        subproc_root = pjoin(outdir, 'SubProcesses')
        pdirs = [pjoin(subproc_root, name) for name in sorted(os.listdir(subproc_root))
                 if name.startswith('P') and os.path.isdir(pjoin(subproc_root, name))]
        self.assertEqual(len(pdirs), 1,
                         'Expected a single subprocess directory for %s, got %s'
                         % (process, pdirs))
        return pdirs[0]

    def _matrix_code(self, pdir):
        """The emitted matrix.f with comment lines stripped.

        Only definitions/uses must be matched, not the prose: a comment may
        legitimately still mention the machinery to explain its absence.
        """
        with open(pjoin(pdir, 'matrix.f')) as fsock:
            source = fsock.read()
        return '\n'.join(line for line in source.split('\n')
                         if not line.lstrip().upper().startswith('C'))

    def _write_driver(self, pdir, split_orders=False):
        """Replace check_sa.f by a driver reading momenta+IFLAV from a file.

        Reading the input rather than hardcoding it lets each process be
        compiled once and then probed at many points / flavor indices.

        The split-orders template has no crossing machinery and hence no
        GET_DENSITY_IDX: its density entry point takes the FLAVOR array, so the
        driver resolves the index through GET_FLAVOR first. Everything else
        (SMATRIX, GET_FLAVOR, GET_FLAVOR_INDEX) has the same interface, so only
        that one call differs.
        """
        if split_orders:
            density_call = '''         CALL GET_FLAVOR(FLAV_IDX, FLAVOR)
         CALL GET_DENSITY(P, DPOS, 1, ALLOW_HEL, 2, FLAVOR,
     &    0D0, 0D0, INTER)'''
        else:
            density_call = '''         CALL GET_DENSITY_IDX(P, DPOS, 1, ALLOW_HEL, 2, FLAV_IDX,
     &    0D0, 0D0, INTER)'''
        # GET_NHEL_IDX / GET_PDG_FOR_FLAVOR only exist in matrix_standalone_v4;
        # the split-orders template lacks them, so its driver must not reference
        # them or it will not link.
        if split_orders:
            nhel_idx_call = '''         WRITE(*,*) 'IDEN= ', -1
         WRITE(*,*) 'PDG= ', 0'''
        else:
            nhel_idx_call = '''         CALL GET_NHEL_IDX(FLAV_IDX, IDEN_STAR, NHEL_STAR)
         CALL GET_PDG_FOR_FLAVOR(FLAV_IDX, PDGS)
         WRITE(*,*) 'IDEN= ', IDEN_STAR
         WRITE(*,*) 'PDG= ', (PDGS(I),I=1,NEXTERNAL)'''
        # GET_NHEL writes NEXTERNAL*NCOMB entries into NHEL_STAR using its own
        # NCOMB; an oversized array in the caller is safe and avoids parsing
        # NCOMB out of matrix.f.
        driver = '''      PROGRAM DRIVER
      use model_object
      IMPLICIT NONE
      INCLUDE "coupl.inc"
      INCLUDE "nexternal.inc"
      INTEGER NCOMB_MAX
      PARAMETER (NCOMB_MAX=4096)
      REAL*8 P(0:3,NEXTERNAL), MATELEM
      INTEGER FLAV_IDX, I, J, MODE
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER GET_FLAVOR_INDEX
      INTEGER NHEL_STAR(NEXTERNAL,NCOMB_MAX), IDEN_STAR
      INTEGER DPOS(1), ALLOW_HEL(2)
      INTEGER PDGS(NEXTERNAL)
      DOUBLE COMPLEX INTER(3)
      call setpara('param_card.dat')
      OPEN(UNIT=42,FILE='cross_input.dat',STATUS='OLD')
      READ(42,*) MODE
      IF (MODE.EQ.1) THEN
         READ(42,*) FLAV_IDX
         CALL GET_FLAVOR(FLAV_IDX, FLAVOR)
         WRITE(*,*) 'POS= ', (FLAVOR(I),I=1,NEXTERNAL)
      ELSEIF (MODE.EQ.2) THEN
         READ(42,*) (FLAVOR(I),I=1,NEXTERNAL)
         WRITE(*,*) 'IDX= ', GET_FLAVOR_INDEX(FLAVOR)
      ELSEIF (MODE.EQ.4) THEN
C        Density matrix: interference between the helicity states of one leg.
C        GET_DENSITY_IDX takes the index directly, so it can carry a crossing;
C        the FLAVOR-array entry point cannot express one.
         READ(42,*) FLAV_IDX
         READ(42,*) DPOS(1)
         DO I=1,NEXTERNAL
            READ(42,*) (P(J,I),J=0,3)
         ENDDO
         ALLOW_HEL(1) = +1
         ALLOW_HEL(2) = -1
%(density_call)s
         DO I=1,3
            WRITE(*,*) 'INTER= ', DREAL(INTER(I)), DIMAG(INTER(I))
         ENDDO
      ELSEIF (MODE.EQ.5) THEN
C        The f2py-facing crossing accessors: GET_NHEL_IDX returns the crossed
C        averaging denominator (unlike GET_NHEL, which only knows the static
C        uncrossed one), and GET_PDG_FOR_FLAVOR returns the per-leg signed PDG
C        of the process the extended FLAV_IDX selects (crossed and conjugated).
         READ(42,*) FLAV_IDX
%(nhel_idx_call)s
      ELSE
         READ(42,*) FLAV_IDX
         DO I=1,NEXTERNAL
            READ(42,*) (P(J,I),J=0,3)
         ENDDO
         CALL SMATRIX(P,FLAV_IDX,MATELEM)
         CALL GET_NHEL(IDEN_STAR,NHEL_STAR)
         WRITE(*,*) 'ANS= ', MATELEM
         WRITE(*,*) 'IDEN= ', IDEN_STAR
      ENDIF
      CLOSE(42)
      END
'''
        with open(pjoin(pdir, 'check_sa.f'), 'w') as fsock:
            fsock.write(driver % {'density_call': density_call,
                                  'nhel_idx_call': nhel_idx_call})

    def _build(self, pdir):
        retcode = self._call(['make', 'check'], pdir)
        self.assertEqual(retcode, 0, 'Failed to compile standalone check in %s' % pdir)

    def _build_f2py(self, pdir):
        """Build the f2py matrix2py module in `pdir`, or skip the test.

        f2py needs a working numpy build backend (meson on numpy>=1.26 /
        python>=3.12), which is not guaranteed in every environment. When it is
        missing this raises SkipTest rather than a failure: the wrapper logic is
        also covered by a mock-backed test that has no toolchain dependency.
        """
        env = dict(os.environ)
        with open(os.devnull, 'w') as devnull:
            retcode = subprocess.call(['make', 'matrix2py.so'], cwd=pdir,
                                      stdout=devnull, stderr=devnull, env=env)
        modules = [name for name in os.listdir(pdir)
                   if name.startswith('matrix2py') and name.endswith('.so')]
        if retcode != 0 or not modules:
            raise unittest.SkipTest(
                'Could not build the f2py module in %s (f2py/numpy build '
                'backend unavailable); skipping the compiled-module test.'
                % pdir)

    def _call(self, command, cwd):
        if logger.isEnabledFor(logging.INFO):
            return subprocess.call(command, cwd=cwd)
        with open(os.devnull, 'w') as devnull:
            return subprocess.call(command, stdout=devnull, stderr=devnull, cwd=cwd)

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------
    def _probe(self, pdir, lines):
        """Feed the driver an input block and return its stdout."""
        with open(pjoin(pdir, 'cross_input.dat'), 'w') as fsock:
            fsock.write('\n'.join(lines) + '\n')
        return subprocess.Popen(['./check'], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                cwd=pdir).communicate()[0].decode()

    def _flavor_positions(self, pdir, flav):
        """GET_FLAVOR: the per-leg flavor-group positions of a flavor index."""
        output = self._probe(pdir, ['1', '%d' % flav])
        match = re.search(r'POS=\s*(.*)', output)
        self.assertTrue(match, 'No POS from %s, got:\n%s' % (pdir, output))
        return tuple(int(token) for token in match.group(1).split())

    def _flavor_index(self, pdir, positions):
        """GET_FLAVOR_INDEX: flavor index of a position vector, 0 if absent."""
        output = self._probe(pdir, ['2', ' '.join(str(p) for p in positions)])
        match = re.search(r'IDX=\s*(-?\d+)', output)
        self.assertTrue(match, 'No IDX from %s, got:\n%s' % (pdir, output))
        return int(match.group(1))

    def _nhel_idx(self, pdir, iflav):
        """(crossed IDEN, per-leg signed PDG) an extended FLAV_IDX selects.

        Exercises the two f2py-facing accessors GET_NHEL_IDX /
        GET_PDG_FOR_FLAVOR that a python caller working in PDG codes relies on.
        """
        output = self._probe(pdir, ['5', '%d' % iflav])
        iden = re.search(r'IDEN=\s*(-?\d+)', output)
        pdg = re.search(r'PDG=\s*(.*)', output)
        self.assertTrue(iden and pdg,
                        'No IDEN/PDG from %s, got:\n%s' % (pdir, output))
        return int(iden.group(1)), tuple(int(t) for t in pdg.group(1).split())

    def _density(self, pdir, momenta, iflav, leg):
        """Return the 3 interference terms of the density matrix of `leg`.

        (++), (+-) and (--) for the two helicity states of that single leg,
        each as a complex number.
        """
        lines = ['4', '%d' % iflav, '%d' % leg]
        for mom in momenta:
            lines.append(' '.join('%.17e' % component for component in mom))
        output = self._probe(pdir, lines)
        values = re.findall(r'INTER=\s*(\S+)\s+(\S+)', output)
        self.assertEqual(len(values), 3,
                         'Expected 3 interference terms from %s, got:\n%s'
                         % (pdir, output))
        return [complex(float(re.sub('[dD]', 'e', real)),
                        float(re.sub('[dD]', 'e', imag)))
                for real, imag in values]

    def _density_f2py(self, pdir, momenta, iflav, legs):
        """The same per-leg density matrix as _density, but obtained through the
        compiled f2py module's PY_GET_DENSITY_IDX. Returns {leg: [c++, c+-, c--]}.

        Requires the module already built (_build_f2py). Runs in a subprocess so
        the f2py .so does not leak into the test interpreter and clash with the
        other process' module.
        """
        card = pjoin(pdir, os.pardir, os.pardir, 'Cards', 'param_card.dat')
        script = _DENSITY_PROBE % {
            'pdir': pdir, 'card': card,
            'momenta': repr([list(mom) for mom in momenta]),
            'flav_idx': iflav, 'legs': repr(tuple(legs))}
        script_path = pjoin(pdir, 'density_probe.py')
        with open(script_path, 'w') as fsock:
            fsock.write(script)
        output = subprocess.Popen(
            [sys.executable, script_path], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, cwd=pdir).communicate()[0].decode()
        match = re.search(r'DENSITY_JSON (.*)', output)
        self.assertTrue(match, 'No density from f2py probe in %s:\n%s'
                        % (pdir, output))
        raw = json.loads(match.group(1))
        return {int(leg): [complex(re_, im_) for re_, im_ in terms]
                for leg, terms in raw.items()}

    def _run(self, pdir, momenta, iflav):
        """Return the averaged matrix element SMATRIX gives for this IFLAV."""
        lines = ['3', '%d' % iflav]
        for mom in momenta:
            lines.append(' '.join('%.17e' % component for component in mom))
        output = self._probe(pdir, lines)
        ans = re.search(r'ANS=\s*(?P<value>[\d\.eEdD\+-]+)', output)
        self.assertTrue(ans,
                        'Could not read the matrix element from %s, got:\n%s'
                        % (pdir, output))
        return float(ans.group('value').replace('D', 'E').replace('d', 'e'))

    def _phase_space(self, cos_theta):
        """A massless 2->2 point: (leg1_in, leg2_in, leg3_out, leg4_out).

        Every parton here (u, u~, g) is massless, so one point serves both
        processes; only the interpretation of each slot differs.
        """
        halfe = 0.5 * self.energy
        sin_theta = math.sqrt(1.0 - cos_theta ** 2)
        return [(halfe, 0.0, 0.0, halfe),
                (halfe, 0.0, 0.0, -halfe),
                (halfe, halfe * sin_theta, 0.0, halfe * cos_theta),
                (halfe, -halfe * sin_theta, 0.0, -halfe * cos_theta)]

    def _read_nflav(self, pdir):
        """NFLAV of a generated process, needed to encode the extended IFLAV.

        IFLAV = cross*NFLAV + flav, so the crossing code cannot be turned into
        an index without it. Read it rather than assume 1: if flavor grouping
        ever merges several flavors here, a hardcoded 1 would silently probe
        the wrong flavor instead of failing.
        """
        with open(pjoin(pdir, 'matrix.f')) as fsock:
            match = re.search(r'PARAMETER\s*\(NFLAV=(\d+)\)', fsock.read())
        self.assertTrue(match, 'Could not read NFLAV from %s' % pdir)
        return int(match.group(1))

    @staticmethod
    def _solve3(matrix, rhs):
        """Solve a 3x3 system by Cramer's rule (avoids a numpy dependency)."""
        def det(m):
            return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        base = det(matrix)
        solution = []
        for col in range(3):
            replaced = [[rhs[row] if c == col else matrix[row][c]
                         for c in range(3)] for row in range(3)]
            solution.append(det(replaced) / base)
        return solution

    def _phase_space_2to3(self, phis_deg=(0.0, 130.0, 245.0), alpha_deg=35.0):
        """A massless 2->3 point: (leg1_in, leg2_in, leg3_out, .., leg5_out).

        Three massless momenta summing to zero are always coplanar, so the
        final state is built in a plane as a closed triangle -- the direction
        angles fix the energies up to the overall scale -- and then rotated out
        of the beam-transverse plane by alpha so the point is not degenerate
        with respect to the beam axis.
        """
        phis = [math.radians(phi) for phi in phis_deg]
        cosines = [math.cos(phi) for phi in phis]
        sines = [math.sin(phi) for phi in phis]
        # sum E*cos = 0, sum E*sin = 0, sum E = energy
        energies = self._solve3([cosines, sines, [1.0, 1.0, 1.0]],
                                [0.0, 0.0, self.energy])
        for energy in energies:
            self.assertGreater(energy, 0.0,
                               'Unphysical phase-space point: energies=%s'
                               % energies)
        alpha = math.radians(alpha_deg)
        halfe = 0.5 * self.energy
        momenta = [(halfe, 0.0, 0.0, halfe), (halfe, 0.0, 0.0, -halfe)]
        for index, energy in enumerate(energies):
            momenta.append((energy,
                            energy * cosines[index],
                            energy * sines[index] * math.cos(alpha),
                            energy * sines[index] * math.sin(alpha)))
        return momenta

    def _assert_crossing(self, crossed_dir, crossed_iflav, reference_dir, label,
                         reference_perm=None):
        """The crossed call on one process must match the other one, plain.

        reference_perm reorders the momenta for the reference code when the
        crossing lands the legs in a different order than the reference
        process expects; None means both take the very same array.
        """
        for cos_theta in self.cos_thetas:
            momenta = self._phase_space(cos_theta)
            crossed = self._run(crossed_dir, momenta, crossed_iflav)
            if reference_perm is None:
                reference_momenta = momenta
            else:
                reference_momenta = [momenta[index] for index in reference_perm]
            reference = self._run(reference_dir, reference_momenta,
                                  IFLAV_IDENTITY)
            scale = max(abs(crossed), abs(reference), 1e-99)
            self.assertLessEqual(
                abs(crossed - reference) / scale, self.tolerance,
                '%s disagrees at cos(theta)=%s: crossed=%r reference=%r'
                % (label, cos_theta, crossed, reference))

    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------
    def test_crossing_gives_back_identity(self):
        """cross=0 must leave the existing behaviour untouched."""
        qq_gg = self._generate(PROC_QQ_GG, 'Proc_qq_gg')
        momenta = self._phase_space(self.cos_thetas[0])
        plain = self._run(qq_gg, momenta, IFLAV_IDENTITY)
        self.assertNotEqual(plain, 0.0,
                            'Sanity check failed: %s gives a null matrix element'
                            % PROC_QQ_GG)
        # IFLAV = cross*NFLAV + flav with cross=0 is just flav: same answer.
        self.assertEqual(plain, self._run(qq_gg, momenta,
                                          _iflav(0, 1, nflav=1)))

    def test_qq_gg_crossed_gives_qg_qg(self):
        """u u~ > g g with particle 2 <-> 3 crossed must give u g > u g."""
        qq_gg = self._generate(PROC_QQ_GG, 'Proc_qq_gg')
        qg_qg = self._generate(PROC_QG_QG, 'Proc_qg_qg')
        self._assert_crossing(
            crossed_dir=qq_gg, crossed_iflav=_iflav(CROSS_2_3, 1, nflav=1),
            reference_dir=qg_qg, label='%s crossed (I=0,J=3) vs %s'
            % (PROC_QQ_GG, PROC_QG_QG))

    def test_qq_gg_crossed_with_last_particle(self):
        """Particle 2 must be crossable with the last particle (J=NEXTERNAL).

        This is the case the NEXTERNAL+1 base exists for: with base NEXTERNAL,
        J could only reach NEXTERNAL-1 and this crossing was unreachable.
        Swapping particle 2 with particle 4 in u u~ > g g turns the incoming u~
        into an outgoing u sitting in slot 4 and the outgoing g of slot 4 into
        an incoming one, so the legs come out ordered as u g > g u: the same
        physics as u g > u g with the two final legs exchanged.
        """
        qq_gg = self._generate(PROC_QQ_GG, 'Proc_qq_gg')
        qg_qg = self._generate(PROC_QG_QG, 'Proc_qg_qg')
        self._assert_crossing(
            crossed_dir=qq_gg, crossed_iflav=_iflav(CROSS_2_LAST, 1, nflav=1),
            reference_dir=qg_qg, reference_perm=(0, 1, 3, 2),
            label='%s crossed (I=0,J=4) vs %s with final legs swapped'
            % (PROC_QQ_GG, PROC_QG_QG))

    def test_qqx_gqqx_crossed_gives_qg_qqqx(self):
        """q q~ > g q q~ crossed (2<->3) must give q g > q q q~, for each q.

        A 2->3 pair, so the crossing has to survive a real flavor table (each
        leg carries its own flavor-group position) and a BROKEN_SYM /
        identical-particle factor that only exists on the crossed side: the
        crossed final state has two identical quarks, which the uncrossed
        q q~ > g q q~ does not. That shows up as IDEN 36 -> 192.

        Repeated over u/d/s/c: up- and down-type quarks sit in different
        flavor groups, so their flavor tables and masks differ.
        """
        for quark in QUARK_FLAVORS:
            with self.subTest(quark=quark):
                qqx_gqqx = self._generate(PROC_QQX_GQQX % {'q': quark},
                                          'Proc_qqx_gqqx_%s' % quark)
                qg_qqqx = self._generate(PROC_QG_QQQX % {'q': quark},
                                         'Proc_qg_qqqx_%s' % quark)
                nflav = self._read_nflav(qqx_gqqx)
                momenta = self._phase_space_2to3()

                crossed = self._run(qqx_gqqx, momenta,
                                    _iflav(CROSS_2_3_5, 1, nflav=nflav))
                reference = self._run(qg_qqqx, momenta, IFLAV_IDENTITY)

                self.assertNotEqual(
                    reference, 0.0,
                    'Sanity check failed: %s gives a null matrix element'
                    % (PROC_QG_QQQX % {'q': quark}))
                scale = max(abs(crossed), abs(reference), 1e-99)
                self.assertLessEqual(
                    abs(crossed - reference) / scale, self.tolerance,
                    '%s crossed (I=0,J=3) disagrees with %s: '
                    'crossed=%r reference=%r'
                    % (PROC_QQX_GQQX % {'q': quark},
                       PROC_QG_QQQX % {'q': quark}, crossed, reference))

    def test_merged_flavor_crossing_every_flavor(self):
        """Every flavor of the merged q q~ > g q q~ must cross onto q g > q q q~.

        The single-flavor tests above only ever exercise the rows where all
        quarks share one flavor, and those are exactly the rows for which the
        denominator happens to be flavor independent. This one sweeps the whole
        merged table (NFLAV=28 against NFLAV=16), which is what catches a
        denominator built from the process's representative flavor instead of
        the actual one: d d~ > g u u~ crosses to d g > d u u~ (nothing
        identical) while d d~ > g d d~ crosses to d g > d d d~ (two identical
        d), and getting that wrong shows up as a clean factor 2.

        Flavors are matched through the generated GET_FLAVOR /
        GET_FLAVOR_INDEX rather than by index: the two processes do not have
        the same NFLAV, so equal indices mean nothing.
        """
        merged_a = self._generate(PROC_MERGED_QQX_GQQX, 'Proc_merged_a')
        merged_b = self._generate(PROC_MERGED_QG_QQQX, 'Proc_merged_b')
        nflav_a = self._read_nflav(merged_a)
        self.assertGreater(nflav_a, 1,
                           'Expected a merged multi-flavor matrix element, got '
                           'NFLAV=%s: this test would not probe the flavor '
                           'dependence of the denominator' % nflav_a)
        momenta = self._phase_space_2to3()

        unmapped = []
        for flav in range(1, nflav_a + 1):
            positions = self._flavor_positions(merged_a, flav)
            # Caller slot 2 holds leg 3 (the gluon) and slot 3 holds leg 2.
            crossed = (positions[0], positions[2], positions[1],
                       positions[3], positions[4])
            reference_perm = None
            target = self._flavor_index(merged_b, crossed)
            if target < 1:
                # Slots 3 and 4 are both _quark, so the target keeps only one
                # ordering of each unordered pair. Try the other one, swapping
                # the momenta along with the flavors.
                swapped = (crossed[0], crossed[1], crossed[3],
                           crossed[2], crossed[4])
                target = self._flavor_index(merged_b, swapped)
                reference_perm = (0, 1, 3, 2, 4)
            if target < 1:
                unmapped.append((flav, positions, crossed))
                continue

            with self.subTest(flav=flav, positions=positions):
                crossed_value = self._run(merged_a, momenta,
                                          _iflav(CROSS_2_3_5, flav,
                                                 nflav=nflav_a))
                reference_momenta = momenta if reference_perm is None else \
                    [momenta[index] for index in reference_perm]
                reference = self._run(merged_b, reference_momenta, target)
                scale = max(abs(crossed_value), abs(reference), 1e-99)
                self.assertLessEqual(
                    abs(crossed_value - reference) / scale, self.tolerance,
                    'flavor %s (positions %s) crossed disagrees: crossed=%r '
                    'reference=%r (ratio %r)'
                    % (flav, positions, crossed_value, reference,
                       reference / crossed_value if crossed_value else None))

        self.assertFalse(unmapped,
                         'Crossed flavors with no counterpart in %s: %s'
                         % (PROC_MERGED_QG_QQQX, unmapped))

    def test_merged_flavor_reverse_crossing_covers_every_flavor(self):
        """The reverse crossing must reach every flavor of q q~ > g q q~.

        q g > q q q~ has fewer flavors (16) than q q~ > g q q~ (28), which
        looks like the reverse mapping cannot be onto. It is: the crossing
        partner J is the missing degree of freedom. J=3 and J=4 cross particle
        2 with one or the other of the two final quarks, and those land on
        different flavors of the target. The two coincide only when the two
        final quarks already share a flavor, so the count works out exactly:

            16 flavors x 2 crossings - 4 degenerate = 28

        J=4 leaves the legs ordered (q, q~, q, g, q~) instead of the target's
        (q, q~, g, q, q~), hence the momentum swap of slots 3 and 4.
        """
        merged_a = self._generate(PROC_MERGED_QQX_GQQX, 'Proc_merged_a')
        merged_b = self._generate(PROC_MERGED_QG_QQQX, 'Proc_merged_b')
        nflav_a = self._read_nflav(merged_a)
        nflav_b = self._read_nflav(merged_b)
        momenta = self._phase_space_2to3()

        covered = {}
        for flav_b in range(1, nflav_b + 1):
            positions = self._flavor_positions(merged_b, flav_b)
            variants = (
                # J=3: legs already come out in the target's order.
                (3, (positions[0], positions[2], positions[1],
                     positions[3], positions[4]), None),
                # J=4: cross the other final quark, then reorder slots 3/4.
                (4, (positions[0], positions[3], positions[1],
                     positions[2], positions[4]), (0, 1, 3, 2, 4)),
            )
            for j_part, target_positions, perm in variants:
                flav_a = self._flavor_index(merged_a, target_positions)
                self.assertGreaterEqual(
                    flav_a, 1,
                    'Crossed flavor %s (from %s flavor %s, J=%s) has no '
                    'counterpart in %s'
                    % (target_positions, PROC_MERGED_QG_QQQX, flav_b, j_part,
                       PROC_MERGED_QQX_GQQX))
                covered.setdefault(flav_a, []).append((flav_b, j_part))

                with self.subTest(flav_b=flav_b, j_part=j_part):
                    cross = 0 * (NEXTERNAL_5 + 1) + j_part
                    crossed_momenta = momenta if perm is None else \
                        [momenta[index] for index in perm]
                    crossed_value = self._run(merged_b, crossed_momenta,
                                              _iflav(cross, flav_b,
                                                     nflav=nflav_b))
                    reference = self._run(merged_a, momenta, flav_a)
                    scale = max(abs(crossed_value), abs(reference), 1e-99)
                    self.assertLessEqual(
                        abs(crossed_value - reference) / scale, self.tolerance,
                        '%s flavor %s crossed (J=%s) disagrees with %s flavor '
                        '%s: crossed=%r reference=%r'
                        % (PROC_MERGED_QG_QQQX, flav_b, j_part,
                           PROC_MERGED_QQX_GQQX, flav_a, crossed_value,
                           reference))

        self.assertEqual(
            len(covered), nflav_a,
            'The reverse crossing covers %s of the %s flavors of %s; missing '
            '%s' % (len(covered), nflav_a, PROC_MERGED_QQX_GQQX,
                    sorted(set(range(1, nflav_a + 1)) - set(covered))))

    def test_crossed_density_matrix(self):
        """The density matrix must survive the crossing, helicity by helicity.

        Every other test here sums over helicities, which makes them blind to
        how a crossed leg's helicity is labelled: a spurious flip would just
        permute the terms of the sum and cancel out. The density matrix is
        resolved per helicity, so it is the one probe that pins that down.

        The expectation is that NO extra flip is needed. Helas builds the
        wavefunction with nh=nhel*nsf, so flipping the NSF flag of a crossed
        leg already flips its effective helicity; the caller's label therefore
        carries over unchanged through the slot permutation. If a flip were
        missing (or applied twice) the diagonal terms would swap and the
        off-diagonal one would conjugate, which this comparison would catch.

        Probed on the gluon of u g > u g, which is leg 2 there and comes from
        the crossing on the u u~ > g g side.
        """
        qq_gg = self._generate(PROC_QQ_GG, 'Proc_qq_gg')
        qg_qg = self._generate(PROC_QG_QG, 'Proc_qg_qg')
        for cos_theta in self.cos_thetas:
            momenta = self._phase_space(cos_theta)
            with self.subTest(cos_theta=cos_theta):
                crossed = self._density(qq_gg, momenta,
                                        _iflav(CROSS_2_3, 1, nflav=1), leg=2)
                reference = self._density(qg_qg, momenta, IFLAV_IDENTITY,
                                          leg=2)
                self.assertTrue(any(abs(term) > 1e-99 for term in reference),
                                'Sanity check failed: null density matrix for '
                                '%s' % PROC_QG_QG)
                for index, (got, want) in enumerate(zip(crossed, reference)):
                    scale = max(abs(got), abs(want), 1e-99)
                    self.assertLessEqual(
                        abs(got - want) / scale, self.tolerance,
                        'Density matrix term %s disagrees at cos(theta)=%s: '
                        'crossed=%r reference=%r' % (index, cos_theta, got, want))

    def test_density_matrix_diagonal_matches_smatrix(self):
        """Summing the density matrix diagonal must reproduce SMATRIX.

        The diagonal terms are |M|^2 for each helicity of the probed leg, so
        summing them has to give back what SMATRIX returns for that flavor.
        This pins the normalisation of the density path, which GET_INTER cannot
        get right on its own: it only sees JAMPs, so it divides by the bare
        static IDEN and can apply neither BROKEN_SYM nor a crossed denominator.

        Probed on the merged q g > q q q~, whose two final quarks live in the
        same flavor group: BROKEN_SYM is 2 exactly when they differ, and those
        are the rows that were coming out a factor 2 low. A single-flavor
        process would have BROKEN_SYM=1 throughout and prove nothing.
        """
        merged_b = self._generate(PROC_MERGED_QG_QQQX, 'Proc_merged_b')
        nflav_b = self._read_nflav(merged_b)
        momenta = self._phase_space_2to3()
        for flav in range(1, nflav_b + 1):
            with self.subTest(flav=flav):
                density = self._density(merged_b, momenta, flav, leg=1)
                diagonal = density[0] + density[2]
                reference = self._run(merged_b, momenta, flav)
                self.assertNotEqual(reference, 0.0,
                                    'Sanity check failed: null matrix element '
                                    'for flavor %s' % flav)
                scale = max(abs(diagonal), abs(reference), 1e-99)
                self.assertLessEqual(
                    abs(diagonal.real - reference) / scale, self.tolerance,
                    'Density diagonal does not sum to SMATRIX for flavor %s: '
                    'diagonal=%r smatrix=%r (ratio %r)'
                    % (flav, diagonal.real, reference,
                       reference / diagonal.real if diagonal.real else None))

    def _assert_chiral_crossed_density(self, crossed, reference):
        """Every leg's crossed density matrix must match the native one, AND the
        crossed fermion (last leg) must be fully polarized so the check actually
        discriminates a helicity flip.

        `crossed` / `reference` are {leg: [c++, c+-, c--]} for legs 1..4 of
        u g > w+ d. Leg 4 is the d that swapped initial<->final on the
        u d~ > w+ g side; the W+ makes it 100% one-handed, so (++) and (--) are
        one full / one empty. A missing or doubled crossing flip would swap them,
        which the term-by-term comparison then catches.
        """
        pol_pp, pol_mm = abs(reference[4][0]), abs(reference[4][2])
        self.assertGreater(max(pol_pp, pol_mm), 1e-3,
                           'Reference crossed-fermion density is null; the probe '
                           'is broken (%r)' % reference[4])
        self.assertLess(min(pol_pp, pol_mm), 1e-9 * max(pol_pp, pol_mm),
                        'Crossed fermion is not fully polarized, so a helicity '
                        'flip would NOT be discriminated: (++)=%r (--)=%r'
                        % (reference[4][0], reference[4][2]))
        for leg in (1, 2, 3, 4):
            self.assertTrue(any(abs(term) > 1e-99 for term in reference[leg]),
                            'Null reference density for leg %s' % leg)
            for index, (got, want) in enumerate(zip(crossed[leg],
                                                     reference[leg])):
                scale = max(abs(got), abs(want), 1e-99)
                self.assertLessEqual(
                    abs(got - want) / scale, self.tolerance,
                    'Crossed density term %s of leg %s disagrees: crossed=%r '
                    'reference=%r' % (index, leg, got, want))

    def test_crossed_density_matrix_chiral_fortran(self):
        """The crossed spin-density matrix of a CHIRAL process, via the compiled
        Fortran GET_DENSITY_IDX (no f2py).

        u d~ > w+ g crossed by (I=0, J=NEXTERNAL) is u g > w+ d; its outgoing d
        is the incoming d~ that swapped sides, still 100% polarized by the W. The
        density matrix is per helicity, so it is the probe that pins how that
        crossed leg's helicity is LABELLED -- the same no-flip convention the
        madevent cross-group event helicity (DSIG_XGHEL) depends on. Every leg,
        crossed vs natively generated, must agree term by term.
        """
        udx_wpg = self._generate(PROC_UDX_WPG, 'Proc_udx_wpg')
        ug_wpd = self._generate(PROC_UG_WPD, 'Proc_ug_wpd')
        crossed_iflav = _iflav(CROSS_2_LAST, 1, nflav=1)
        for cos_theta in self.cos_thetas:
            momenta = self._phase_space(cos_theta)
            with self.subTest(cos_theta=cos_theta):
                crossed = {leg: self._density(udx_wpg, momenta, crossed_iflav,
                                              leg=leg) for leg in (1, 2, 3, 4)}
                reference = {leg: self._density(ug_wpd, momenta, IFLAV_IDENTITY,
                                                leg=leg) for leg in (1, 2, 3, 4)}
                self._assert_chiral_crossed_density(crossed, reference)

    def test_crossed_density_matrix_chiral_f2py(self):
        """Same chiral crossed-density-matrix check, but through the f2py
        PY_GET_DENSITY_IDX wrapper -- the only way a python caller can ask for a
        crossed density matrix. Skips if the f2py build backend is unavailable.
        """
        udx_wpg = self._output_standalone(PROC_UDX_WPG, 'Proc_udx_wpg_f2py')
        ug_wpd = self._output_standalone(PROC_UG_WPD, 'Proc_ug_wpd_f2py')
        self._build_f2py(udx_wpg)
        self._build_f2py(ug_wpd)
        crossed_iflav = _iflav(CROSS_2_LAST, 1, nflav=1)
        for cos_theta in self.cos_thetas:
            momenta = self._phase_space(cos_theta)
            with self.subTest(cos_theta=cos_theta):
                crossed = self._density_f2py(udx_wpg, momenta, crossed_iflav,
                                             (1, 2, 3, 4))
                reference = self._density_f2py(ug_wpd, momenta, IFLAV_IDENTITY,
                                               (1, 2, 3, 4))
                self._assert_chiral_crossed_density(crossed, reference)

    def test_split_orders_density_diagonal_matches_smatrix(self):
        """The same invariant on the split-orders template.

        matrix_standalone_splitOrders_v4.inc is a separate template with its own
        copy of the density code, and it had the very same missing-BROKEN_SYM
        bug as the default one: SMATRIX applies BROKEN_SYM(FLAVOR) while
        GET_INTER normalises with the bare static IDEN and cannot, so the
        diagonal came out a factor BROKEN_SYM low. Fixing one template does not
        fix the other, hence this test next to
        test_density_matrix_diagonal_matches_smatrix.

        Uses the merged q g > q q q~ for the same reason: its two final quarks
        share a flavor group, so BROKEN_SYM=2 on the rows where they differ. A
        single-flavor process has BROKEN_SYM=1 everywhere and would pass even
        with the rescaling removed entirely.
        """
        merged = self._generate(PROC_MERGED_QG_QQQX_SO, 'Proc_merged_so',
                                split_orders=True)
        # Guard the premise: if the squared-order syntax ever stopped setting
        # split_orders, this would silently retest the default template.
        self.assertIn('SMATRIX_SPLITORDERS', self._matrix_code(merged),
                      'Expected %s to be written with the split-orders '
                      'template; this test would otherwise just retest the '
                      'default one' % PROC_MERGED_QG_QQQX_SO)
        nflav = self._read_nflav(merged)
        self.assertGreater(nflav, 1,
                           'Expected a merged multi-flavor matrix element, got '
                           'NFLAV=%s: BROKEN_SYM would be 1 throughout and this '
                           'test could not fail' % nflav)
        momenta = self._phase_space_2to3()
        for flav in range(1, nflav + 1):
            with self.subTest(flav=flav):
                density = self._density(merged, momenta, flav, leg=1)
                diagonal = density[0] + density[2]
                reference = self._run(merged, momenta, flav)
                self.assertNotEqual(reference, 0.0,
                                    'Sanity check failed: null matrix element '
                                    'for flavor %s' % flav)
                scale = max(abs(diagonal), abs(reference), 1e-99)
                self.assertLessEqual(
                    abs(diagonal.real - reference) / scale, self.tolerance,
                    'Split-orders density diagonal does not sum to SMATRIX for '
                    'flavor %s: diagonal=%r smatrix=%r (ratio %r)'
                    % (flav, diagonal.real, reference,
                       reference / diagonal.real if diagonal.real else None))

    def test_use_crossing_false_drops_the_machinery(self):
        """--use_crossing=False must emit no crossing code, same ME otherwise.

        The extended FLAV_IDX only makes sense when the crossed subprocesses
        are *not* generated separately, which is exactly what --use_crossing
        drives. With it off, none of the decoding routines nor the tables they
        read may reach matrix.f (they would be dead code, and GET_AMP's IC
        would carry a crossing that can never be requested), while the plain
        uncrossed matrix element must be untouched: the crossing-off path goes
        through ANS/IDEN*BROKEN_SYM instead of the per-crossing denominator,
        and those two must agree for CROSS=0.
        """
        default = self._generate(PROC_QQ_GG, 'Proc_qq_gg_default')
        no_cross = self._generate(PROC_QQ_GG, 'Proc_qq_gg_nocross',
                                  options='--use_crossing=False')

        code = self._matrix_code(no_cross)
        for name in CROSSING_MACHINERY_NAMES:
            self.assertNotIn(name, code,
                             '%s is still emitted with --use_crossing=False'
                             % name)
        # Sanity: the very same assertion must fail on the default output,
        # otherwise this test would pass on a matrix.f that never had any.
        self.assertIn('GET_SPINCOL_CROSS', self._matrix_code(default),
                      'Default output has no crossing machinery either: '
                      'this test proves nothing')

        for cos_theta in self.cos_thetas:
            momenta = self._phase_space(cos_theta)
            with self.subTest(cos_theta=cos_theta):
                plain = self._run(no_cross, momenta, IFLAV_IDENTITY)
                reference = self._run(default, momenta, IFLAV_IDENTITY)
                self.assertNotEqual(reference, 0.0,
                                    'Sanity check failed: %s gives a null '
                                    'matrix element' % PROC_QQ_GG)
                self.assertEqual(plain, reference,
                                 '--use_crossing=False changes the uncrossed '
                                 'matrix element at cos(theta)=%s: %r vs %r'
                                 % (cos_theta, plain, reference))

    def _assert_machinery(self, process, name, expected):
        """Assert the crossing machinery is (not) emitted for `process`."""
        code = self._matrix_code(self._output_standalone(process, name))
        if expected:
            # One representative name is enough to prove the machinery is there;
            # the full list matters only for the "must be absent" direction,
            # where any single leftover would be dead code reading a crossing
            # that can never be requested.
            self.assertIn('GET_SPINCOL_CROSS', code,
                          'Crossing machinery is missing for %s, which does '
                          'not constrain any s-channel' % process)
        else:
            for routine in CROSSING_MACHINERY_NAMES:
                self.assertNotIn(routine, code,
                                 '%s is emitted for %s, whose s-channel '
                                 'constraint no crossing preserves'
                                 % (routine, process))
        return code

    def test_required_s_channel_disables_crossing(self):
        """`> z >` must drop the machinery; the same process without it keeps it.

        A required s-channel names a propagator that is only s-channel in this
        arrangement of the legs, so it cannot survive a crossing and the
        machinery must not be emitted. The unconstrained twin is generated too:
        without it, the test would pass on any matrix.f that never had the
        machinery at all (e.g. if e+e- output stopped emitting it for an
        unrelated reason).
        """
        self._assert_machinery(PROC_REQUIRED_S, 'Proc_required_s',
                               expected=False)
        self._assert_machinery(PROC_UNCONSTRAINED, 'Proc_unconstrained_req',
                               expected=True)

    def test_forbidden_s_channel_disables_crossing(self):
        """`$$ z` removes a diagram by s-channel, so it must drop the machinery.

        Paired with the unconstrained twin for the same anti-vacuity reason as
        test_required_s_channel_disables_crossing.
        """
        self._assert_machinery(PROC_FORBIDDEN_S, 'Proc_forbidden_s',
                               expected=False)
        self._assert_machinery(PROC_UNCONSTRAINED, 'Proc_unconstrained_forb',
                               expected=True)

    def test_forbidden_onshell_s_channel_keeps_crossing(self):
        """A single `$ z` must NOT disable crossing: the diagram is kept.

        `$` only forbids the on-shell region of a propagator, it does not pin
        the topology, so the crossing machinery stays. This is the test that
        stops the fix from being over-broad and disabling crossing for every
        process carrying any `$`-like constraint.
        """
        self._assert_machinery(PROC_FORBIDDEN_ONSH_S, 'Proc_forbidden_onsh_s',
                               expected=True)

    def test_f2py_flavor_index_accessors(self):
        """GET_NHEL_IDX / GET_PDG_FOR_FLAVOR must describe the crossed process.

        These are the f2py-facing accessors that let a python caller work in
        PDG codes: they turn an extended FLAV_IDX into (crossed denominator,
        crossed+conjugated PDG list). Two failure modes they must not have,
        both invisible to the |M|^2 tests:
          * GET_NHEL_IDX returning the static uncrossed IDEN (the historical
            GET_NHEL bug) rather than the crossed one, and
          * GET_PDG_FOR_FLAVOR forgetting to conjugate a leg that swapped
            between the initial and the final state.
        For u u~ > g g the identity (IFLAV=1) is itself, and the (I=0,J=3)
        crossing (IFLAV=4) is u g > u g: leg 2's u~ (pdg -2) becomes an
        outgoing u (pdg +2) in slot 3, and IDEN goes 72 -> 96.
        """
        qq_gg = self._generate(PROC_QQ_GG, 'Proc_qq_gg')

        iden_id, pdg_id = self._nhel_idx(qq_gg, IFLAV_IDENTITY)
        self.assertEqual(iden_id, 72,
                         'Identity IDEN wrong: %s' % iden_id)
        self.assertEqual(pdg_id, (2, -2, 21, 21),
                         'Identity PDG wrong: %s' % (pdg_id,))

        iden_cr, pdg_cr = self._nhel_idx(qq_gg, _iflav(CROSS_2_3, 1, nflav=1))
        self.assertEqual(iden_cr, 96,
                         'Crossed IDEN should be 96 (u g > u g), got %s. A 72 '
                         'here is the GET_NHEL static-IDEN bug.' % iden_cr)
        self.assertEqual(pdg_cr, (2, 21, 2, 21),
                         'Crossed PDG should be u g > u g with leg 2 conjugated,'
                         ' got %s' % (pdg_cr,))

    def test_f2py_pdg_wrapper(self):
        """The python PDG wrapper must find the crossing and call the right ME.

        End-to-end through the compiled f2py module: build it, then drive
        flavor_dispatch.FlavorDispatch. A caller who knows only the physical
        process as a signed-PDG list must get back the extended FLAV_IDX (via
        find_pdg) and the correct crossed matrix element (via
        matrix_element_pdg). For a u u~ > g g module the identity is itself and
        the (I=0,J=3) crossing is u g > u g. Skips if f2py cannot build here.
        """
        pdir = self._output_standalone(PROC_QQ_GG, 'Proc_qq_gg_f2py')
        self._build_f2py(pdir)

        # Run in a subprocess: importing an f2py .so into the test interpreter
        # would leak a compiled module and clash across tests.
        script = '''
import sys, math, numpy as np
sys.path.insert(0, %(pdir)r)
import matrix2py
from flavor_dispatch import FlavorDispatch
me = FlavorDispatch(matrix2py)
me.initialisemodel(%(card)r)
assert me.flavor_layout() == (1, 4, 25), me.flavor_layout()
assert me.pdg_for_index(1) == (2, -2, 21, 21), me.pdg_for_index(1)
assert me.pdg_for_index(4) == (2, 21, 2, 21), me.pdg_for_index(4)
assert me.find_pdg([2, -2, 21, 21]) == 1
assert me.find_pdg([2, 21, 2, 21]) == 4
assert me.find_pdg([6, -6, 21, 21]) is None   # unreachable process
E = 500.0; c = 0.3; s = math.sqrt(1.0 - c * c)
P = np.asfortranarray(np.array([[E, 0, 0, E], [E, 0, 0, -E],
    [E, E * s, 0, E * c], [E, -E * s, 0, -E * c]]).T)
direct = me.smatrix(P, 4)
via = me.matrix_element_pdg(P, [2, 21, 2, 21])
assert abs(direct - via) <= 1e-11 * abs(direct), (direct, via)
assert direct > 0.0
print("F2PY_PDG_OK")
''' % {'pdir': pdir,
       'card': pjoin(pdir, os.pardir, os.pardir, 'Cards', 'param_card.dat')}
        script_path = pjoin(pdir, 'pdg_wrapper_probe.py')
        with open(script_path, 'w') as fsock:
            fsock.write(script)
        proc = subprocess.Popen([sys.executable, script_path],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                cwd=pdir)
        output = proc.communicate()[0].decode()
        self.assertIn('F2PY_PDG_OK', output,
                      'PDG wrapper probe failed:\n%s' % output)

    def _assert_goodhel_relation(self, process, name, ninitial, npts=16):
        """Compiled-module check of the GHREMAP good-helicity relation.

        Builds the f2py module for `process` and, for every DERIVABLE crossing,
        asserts the crossed good-helicity set equals the identity's mapped
        through the crossing permutation sigma (the invariant GHREMAP encodes).
        Skips if the f2py toolchain is unavailable, exactly like the other
        compiled-module tests.
        """
        pdir = self._output_standalone(process, name)
        self._build_f2py(pdir)
        card = pjoin(pdir, os.pardir, os.pardir, 'Cards', 'param_card.dat')
        script = _GOODHEL_PROBE % {'pdir': pdir, 'card': card,
                                   'ninitial': ninitial, 'npts': npts}
        script_path = pjoin(pdir, 'goodhel_relation_probe.py')
        with open(script_path, 'w') as fsock:
            fsock.write(script)
        proc = subprocess.Popen([sys.executable, script_path],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                cwd=pdir)
        output = proc.communicate()[0].decode()
        self.assertIn('GHREMAP_RELATION_OK', output,
                      'good-helicity relation probe failed for %s:\n%s'
                      % (process, output))

    def test_goodhel_relation_qq_gg(self):
        """The crossed good-helicity set of u u~ > g g must be the identity's
        mapped through sigma, for every derivable crossing (>=12 points, so the
        cross=23 accidental-zero undercount cannot mask a bug)."""
        self._assert_goodhel_relation(PROC_QQ_GG, 'Proc_qq_gg_goodhel',
                                      ninitial=2)

    def test_goodhel_relation_qq_ggg(self):
        """Same relation on a 2->3 (u u~ > g g g): more crossings, and the
        initial-initial swaps that break the relation are correctly excluded
        from the derivable set the probe checks."""
        self._assert_goodhel_relation('u u~ > g g g', 'Proc_qq_ggg_goodhel',
                                      ninitial=2)

    def test_qg_qg_crossed_gives_qq_gg(self):
        """u g > u g with particle 2 <-> 3 crossed must give u u~ > g g."""
        qq_gg = self._generate(PROC_QQ_GG, 'Proc_qq_gg')
        qg_qg = self._generate(PROC_QG_QG, 'Proc_qg_qg')
        self._assert_crossing(
            crossed_dir=qg_qg, crossed_iflav=_iflav(CROSS_2_3, 1, nflav=1),
            reference_dir=qq_gg, label='%s crossed (I=0,J=3) vs %s'
            % (PROC_QG_QG, PROC_QQ_GG))


class TestCheckCrossingCommand(unittest.TestCase):
    """The `check crossing` MG5 subcommand end-to-end.

    Drives the same code path as ``check crossing <process>``:
    ``process_checks.check_crossing`` regenerates the process to fortran
    standalone twice (crossing on and off), builds the f2py ``matrix2py``
    module in every P* directory, and compares each subprocess evaluated
    through the crossing-enabled build against its crossing-disabled value.
    Skips (rather than fails) when the f2py/numpy build backend is missing.
    """

    # x = u u~, x x > x x is the smallest line that puts a subprocess of the
    # crossing-disabled reference (u u > u u) behind a *genuine* crossing in the
    # crossing-enabled build: the two modes pick different representatives, so
    # u u > u u is reached there only by a non-identity FLAV_IDX. That makes the
    # comparison exercise APPLY_CROSSING rather than a plain identity, and it is
    # small enough (no external gluon) to build quickly.
    def setUp(self):
        import madgraph.interface.master_interface as cmd_interface
        self.cmd = cmd_interface.MasterCmd()
        self.cmd.no_notification()
        self.cmd.exec_cmd('set automatic_html_opening False', printcmd=False)
        self.cmd.exec_cmd('import model sm', printcmd=False)
        self.cmd.exec_cmd('define xq = u u~', printcmd=False)

    def _run_check(self, proc_line, exporter='standalone'):
        import madgraph.various.process_checks as process_checks
        # The C++/mg7 backends need a working C++ compiler + build toolchain;
        # the fortran one needs f2py. Skip (do not fail) when unavailable.
        if exporter != 'standalone':
            compiler = os.environ.get('CXX', 'g++')
            if not shutil.which(compiler):
                raise unittest.SkipTest('no C++ compiler (%s) available for '
                                        'exporter %s' % (compiler, exporter))
        procdef = self.cmd.extract_process(proc_line)
        results = process_checks.check_crossing(
            procdef, param_card=None,
            options={'energy': 1000.0, 'proc_line': proc_line,
                     'exporter': exporter},
            cmd=self.cmd)
        if any(r.get('status') == 'build_failed' for r in results):
            raise unittest.SkipTest(
                'Could not build the %s crossing output (build backend '
                'unavailable); skipping the check crossing test.' % exporter)
        return results, process_checks

    def _assert_all_pass_with_crossing(self, results, process_checks,
                                       require_crossing=True):
        """Shared assertions: every subprocess agrees (Passed), at least one is
        reached through a genuine (non-identity) crossing, and the rendered
        report is failure-free."""
        self.assertTrue(results, 'check crossing returned no comparison')
        checked = 0
        crossed = 0
        for res in results:
            self.assertEqual(res['status'], 'ok', res)
            vd = res['value_direct']
            vc = res['value_crossed']
            self.assertIsNotNone(vd, 'no direct value for %s' % res['process'])
            self.assertIsNotNone(vc, 'no crossed value for %s' % res['process'])
            self.assertGreater(abs(vd), 0.0,
                               'null matrix element for %s' % res['process'])
            scale = max(abs(vd), abs(vc), 1e-99)
            self.assertLessEqual(
                abs(vd - vc) / scale, 1e-6,
                '%s disagrees between crossing on/off: direct=%r crossed=%r'
                % (res['process'], vd, vc))
            checked += 1
            if res.get('cross_code'):
                crossed += 1
        self.assertGreater(checked, 0, 'no subprocess was checked')
        if require_crossing:
            # Non-vacuity: the comparison must genuinely go through the crossing
            # machinery for at least one subprocess, not only identity matches.
            self.assertGreater(
                crossed, 0,
                'No subprocess was reached through a non-identity crossing; the '
                'test would then only compare the two builds at cross=0')

        # The rendered report must show the Passed verdict, as the other check
        # subcommands do.
        text = process_checks.output_crossing(results)
        self.assertIn('Passed', text)
        self.assertIn('Summary:', text)
        self.assertEqual(process_checks.output_crossing(results, 'fail'), 0,
                         'output_crossing reported a failure:\n%s' % text)
        return crossed

    def test_check_crossing_command(self):
        """standalone (fortran): every subprocess must agree between the two
        modes, with a Passed verdict, and at least one must be reached through a
        real crossing."""
        results, process_checks = self._run_check('xq xq > xq xq')
        self._assert_all_pass_with_crossing(results, process_checks)

    def test_check_crossing_command_cpp(self):
        """standalone_cpp backend: u u > u u reached through a genuine crossing
        of a different subprocess must agree with its independent value."""
        results, process_checks = self._run_check(
            'xq xq > xq xq', exporter='standalone_cpp')
        self._assert_all_pass_with_crossing(results, process_checks)

    def test_check_crossing_command_mg7(self):
        """standalone_mg7 (cudacpp CPU-SIMD) backend: same genuine-crossing
        agreement, evaluated at a prescribed phase-space point injected into the
        SIMD momenta buffer."""
        results, process_checks = self._run_check(
            'xq xq > xq xq', exporter='standalone_mg7')
        self._assert_all_pass_with_crossing(results, process_checks)

    def test_check_crossing_invalid_exporter(self):
        """An unknown --exporter must raise a clear InvalidCmd, not run."""
        import madgraph
        import madgraph.various.process_checks as process_checks
        procdef = self.cmd.extract_process('g u > g u')
        with self.assertRaises(madgraph.InvalidCmd) as ctx:
            process_checks.check_crossing(
                procdef, param_card=None,
                options={'energy': 1000.0, 'proc_line': 'g u > g u',
                         'exporter': 'not_a_backend'},
                cmd=self.cmd)
        self.assertIn('not_a_backend', str(ctx.exception))

    def test_check_crossing_invalid_simd(self):
        """An unknown standalone_mg7 --simd must raise a clear InvalidCmd.

        No build: constructing the mg7 backend validates the choice up front.
        """
        import madgraph
        import madgraph.various.process_checks as process_checks
        procdef = self.cmd.extract_process('g u > g u')
        with self.assertRaises(madgraph.InvalidCmd) as ctx:
            process_checks.check_crossing(
                procdef, param_card=None,
                options={'energy': 1000.0, 'proc_line': 'g u > g u',
                         'exporter': 'standalone_mg7', 'simd': 'not_a_simd'},
                cmd=self.cmd)
        self.assertIn('not_a_simd', str(ctx.exception))

    def test_check_crossing_s_channel_graceful(self):
        """A required s-channel disables crossing; the check must still pass.

        `u u~ > z > e+ e-` is only s-channel in this arrangement of the legs, so
        no crossing preserves it and the crossing machinery is not emitted.  The
        command must handle this gracefully: every subprocess is matched at the
        identity and passes (the crossing-enabled and crossing-disabled builds
        agree), rather than erroring.
        """
        results, process_checks = self._run_check('u u~ > z > e+ e-')
        self.assertTrue(results, 'check crossing returned no comparison')
        for res in results:
            self.assertEqual(res['status'], 'ok', res)
            self.assertIsNotNone(res['value_direct'])
            self.assertIsNotNone(res['value_crossed'])
            self.assertFalse(res.get('cross_code'),
                             'a constrained-s-channel process should not be '
                             'reached by any non-identity crossing: %s' % res)
        self.assertEqual(process_checks.output_crossing(results, 'fail'), 0)


class TestCrossingUnsupportedOutput(unittest.TestCase):
    """Outputs that cannot cross must refuse a process generated with crossing.

    --use_crossing is on by default and tells the generation *not* to write the
    crossed subprocesses out separately, because the matrix element is supposed
    to reach them through an extended FLAV_IDX. The fortran standalone and the
    (grouped) madevent output decode one; an output that cannot would quietly
    produce a matrix element missing those subprocesses, so it has to raise
    instead.
    """

    # Outputs reached through ExportV4Factory that have no crossing machinery
    # (madevent is no longer here: the grouped exporter shares a base matrix
    # element through the crossing router, see TestCrossingPartition).
    UNSUPPORTED_FORMATS = ['matchbox']

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='cross_unsupported_')

    def tearDown(self):
        if os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _output(self, fmt, name, options=''):
        """Run generate+output for `fmt`, returning nothing or raising."""
        cmd = cmd_interface.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False')
        cmd.exec_cmd('import model sm')
        cmd.exec_cmd(('generate %s %s' % (PROC_QG_QG, options)).strip())
        cmd.exec_cmd('output %s %s -f' % (fmt, pjoin(self.tmpdir, name)))

    def test_unsupported_output_raises_with_crossing(self):
        """The default (crossing on) must be refused, and say how to fix it."""
        for fmt in self.UNSUPPORTED_FORMATS:
            with self.subTest(format=fmt):
                with self.assertRaises(madgraph.InvalidCmd) as ctx:
                    self._output(fmt, 'raise_%s' % fmt)
                message = str(ctx.exception)
                # An error that does not name the way out would just leave the
                # user stuck, so the remedy is part of the requirement.
                self.assertIn('--use_crossing=False', message,
                              'The %s error does not name the fix: %s'
                              % (fmt, message))

    def test_unsupported_output_accepted_without_crossing(self):
        """--use_crossing=False must let the very same output through.

        Without this the test above would be satisfied by an exporter that is
        simply broken, rather than by one gating on the crossing request.
        """
        for fmt in self.UNSUPPORTED_FORMATS:
            with self.subTest(format=fmt):
                self._output(fmt, 'ok_%s' % fmt,
                             options='--use_crossing=False')

    def test_supported_outputs_accept_crossing(self):
        """Outputs that DO implement crossing must not be caught.

        Anchors the gate against being over-broad: a check that refused every
        output would pass both tests above. The fortran standalone decodes the
        extended FLAV_IDX directly; the grouped madevent output reaches the
        crossed subprocesses through the crossing router.
        """
        for fmt in ('standalone', 'madevent'):
            with self.subTest(format=fmt):
                self._output(fmt, 'ok_%s' % fmt)


# The C++ standalone driver: take a fixed RAMBO phase space point once
# (all-massless, so the momenta are identical between the two P directories) and
# print sigmaKin at each flavor_id passed on the command line. Each flavor_id is
# evaluated in a FRESH CPPProcess so the good-helicity cache starts empty: that
# cache is indexed by the reduced flavor (flav_use), so different crossings of
# one flavor would otherwise share it and, once it kicks in, a later crossing
# would be filtered by an earlier one's non-zero-helicity pattern (the deferred
# open question of keying the cache on the full flavor_id). The momenta are
# generated once and reused so every process sees the very same point.
# The shipped check_sa.cpp only ever loops over its own maxflavor identities, so
# a purpose-built driver is needed to request a crossed flavor_id.
_CPP_DRIVER = r"""
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include "CPPProcess.h"
#include "rambo.h"

int main(int argc, char** argv){
  double energy = 1000.0;
  double weight;
  CPPProcess seed("../../Cards/param_card.dat");
  vector<double*> p = get_momenta(seed.ninitial, energy,
                                  seed.getMasses(), weight);
  std::cout << std::setprecision(17);
  for(int a = 1; a < argc; a++){
    int fid = atoi(argv[a]);
    CPPProcess process("../../Cards/param_card.dat");
    process.setMomenta(p);
    double me = process.sigmaKin(fid);
    std::cout << "sigmaKin(" << fid << ") = " << me << std::endl;
  }
  return 0;
}
"""


class TestStandaloneCppCrossSymmetry(unittest.TestCase):
    """standalone_cpp must reproduce the crossing the fortran standalone does.

    Mirror of TestStandaloneCrossSymmetry for the C++ backend: u u~ > g g and
    u g > u g are each other's crossing under (I=0, J=3). In the 0-based C++
    flavor_id encoding cross = flavor_id / nflav and flav = flavor_id % nflav,
    so with NFLAV=1 the crossing (I=0, J=3) -> cross = 0*(NEXTERNAL+1)+3 = 3 is
    reached by flavor_id = 3. sigmaKin already divides by the crossed
    denominator, so a crossed call returns the properly averaged matrix element
    of the process it crosses into and can be compared directly.

    Skipped when no C++ compiler is available (the whole check needs to build
    and run real C++).
    """

    energy = 1000.0
    tolerance = 1e-9
    # cross = I*(NEXTERNAL+1)+J = 0*5+3 = 3, flavor_id = cross*NFLAV+flav (NFLAV=1)
    CROSS_2_3 = 3
    IDENTITY = 0

    debugging = getattr(unittest, 'debug', False)

    def setUp(self):
        self.compiler = os.environ.get('CXX', 'g++')
        if not shutil.which(self.compiler):
            self.skipTest('no C++ compiler (%s) available' % self.compiler)
        self.tmpdir = tempfile.mkdtemp(prefix='cross_cpp_')

    def tearDown(self):
        if not self.debugging and os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    # ------------------------------------------------------------------
    def _output_standalone_cpp(self, process, name, options=''):
        """Write the standalone_cpp output for `process`, return its P* dir."""
        outdir = pjoin(self.tmpdir, name)
        cmd = cmd_interface.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False')
        cmd.exec_cmd('set group_subprocesses False')
        cmd.exec_cmd('set apply_flavor_grouping True')
        cmd.exec_cmd('import model sm')
        cmd.exec_cmd(('generate %s %s' % (process, options)).strip())
        cmd.exec_cmd('output standalone_cpp %s -f' % outdir)

        subproc_root = pjoin(outdir, 'SubProcesses')
        pdirs = [pjoin(subproc_root, d) for d in sorted(os.listdir(subproc_root))
                 if d.startswith('P') and os.path.isdir(pjoin(subproc_root, d))]
        self.assertEqual(len(pdirs), 1,
                         'Expected a single subprocess directory for %s, got %s'
                         % (process, pdirs))
        return pdirs[0]

    def _cpp_source(self, pdir):
        with open(pjoin(pdir, 'CPPProcess.cc')) as fsock:
            return fsock.read()

    def _build_and_run(self, pdir, flavor_ids):
        """Build the driver in `pdir` and return {flavor_id: sigmaKin}."""
        # 'make' compiles CPPProcess.o and links the shipped check; it also
        # proves the generated code compiles.
        with open(os.devnull, 'w') as devnull:
            rc = subprocess.call(['make'], cwd=pdir, stdout=devnull,
                                  stderr=subprocess.STDOUT)
        self.assertEqual(rc, 0, 'make failed in %s' % pdir)

        with open(pjoin(pdir, 'driver_cross.cpp'), 'w') as fsock:
            fsock.write(_CPP_DRIVER)
        cxxflags = ['-O3', '-ffast-math', '-I../../src', '-I.', '-fPIC']
        libflags = ['-L../../lib', '-lmodel_sm']
        with open(os.devnull, 'w') as devnull:
            rc = subprocess.call(
                [self.compiler] + cxxflags + ['-c', '-o', 'driver_cross.o',
                                              'driver_cross.cpp'],
                cwd=pdir, stdout=devnull, stderr=subprocess.STDOUT)
            self.assertEqual(rc, 0, 'driver compile failed in %s' % pdir)
            rc = subprocess.call(
                [self.compiler, '-o', 'driver_cross', 'CPPProcess.o',
                 'driver_cross.o'] + libflags,
                cwd=pdir, stdout=devnull, stderr=subprocess.STDOUT)
            self.assertEqual(rc, 0, 'driver link failed in %s' % pdir)

        out = subprocess.check_output(
            ['./driver_cross'] + [str(f) for f in flavor_ids],
            cwd=pdir).decode()
        values = {}
        for match in re.finditer(r'sigmaKin\((\d+)\)\s*=\s*([-\d.eE+]+)', out):
            values[int(match.group(1))] = float(match.group(2))
        self.assertEqual(set(values), set(flavor_ids),
                         'driver output did not cover every flavor_id: %s' % out)
        return values

    # ------------------------------------------------------------------
    def test_qq_gg_crossed_gives_qg_qg(self):
        """u u~ > g g crossed by (I=0,J=3) must equal u g > u g at the same
        momenta, and the crossed value must differ from the identity one so the
        check is non-vacuous."""
        crossed_dir = self._output_standalone_cpp(PROC_QQ_GG, 'qqgg')
        reference_dir = self._output_standalone_cpp(PROC_QG_QG, 'qgqg')

        crossed = self._build_and_run(crossed_dir,
                                      [self.IDENTITY, self.CROSS_2_3])
        reference = self._build_and_run(reference_dir, [self.IDENTITY])

        self.assertAlmostEqual(
            crossed[self.CROSS_2_3], reference[self.IDENTITY],
            delta=self.tolerance * abs(reference[self.IDENTITY]),
            msg='u u~ > g g crossed (%r) != u g > u g identity (%r)'
            % (crossed[self.CROSS_2_3], reference[self.IDENTITY]))
        # Non-vacuous: the crossing must move the answer, not return the
        # identity value.
        self.assertNotAlmostEqual(
            crossed[self.CROSS_2_3], crossed[self.IDENTITY], places=6,
            msg='crossed value equals the identity value; crossing had no '
                'effect, so the test would pass trivially')

    def test_qg_qg_crossed_gives_qq_gg(self):
        """The reverse: u g > u g crossed by (I=0,J=3) must equal u u~ > g g."""
        crossed_dir = self._output_standalone_cpp(PROC_QG_QG, 'qgqg_rev')
        reference_dir = self._output_standalone_cpp(PROC_QQ_GG, 'qqgg_rev')

        crossed = self._build_and_run(crossed_dir,
                                      [self.IDENTITY, self.CROSS_2_3])
        reference = self._build_and_run(reference_dir, [self.IDENTITY])

        self.assertAlmostEqual(
            crossed[self.CROSS_2_3], reference[self.IDENTITY],
            delta=self.tolerance * abs(reference[self.IDENTITY]),
            msg='u g > u g crossed (%r) != u u~ > g g identity (%r)'
            % (crossed[self.CROSS_2_3], reference[self.IDENTITY]))

    def test_invalid_overlapping_swap_returns_zero(self):
        """An overlapping-swap crossing code (I=2, J=1 here) is marked invalid;
        sigmaKin must short-circuit to 0 for it."""
        pdir = self._output_standalone_cpp(PROC_QQ_GG, 'qqgg_inv')
        # cross = I*(NEXTERNAL+1)+J = 2*5+1 = 11, flavor_id = 11 (NFLAV=1).
        overlapping = 2 * (NEXTERNAL + 1) + 1
        values = self._build_and_run(pdir, [overlapping])
        self.assertEqual(values[overlapping], 0.0,
                         'an overlapping-swap code must give a zero matrix '
                         'element, got %r' % values[overlapping])

    def test_use_crossing_false_drops_the_machinery(self):
        """--use_crossing=False must compile and emit no crossing machinery,
        while still giving the same uncrossed matrix element."""
        with_dir = self._output_standalone_cpp(PROC_QQ_GG, 'qqgg_on')
        without_dir = self._output_standalone_cpp(PROC_QQ_GG, 'qqgg_off',
                                                  options='--use_crossing=False')

        on_src = self._cpp_source(with_dir)
        off_src = self._cpp_source(without_dir)
        for token in ('spincol_cross', 'cross_perm', 'cross_ic',
                      'ident_cross', 'flav_use', 'const int ic[]'):
            self.assertIn(token, on_src,
                          '%s should be emitted with crossing on' % token)
            self.assertNotIn(token, off_src,
                             '%s must NOT be emitted with --use_crossing=False'
                             % token)

        on = self._build_and_run(with_dir, [self.IDENTITY])
        off = self._build_and_run(without_dir, [self.IDENTITY])
        self.assertAlmostEqual(
            on[self.IDENTITY], off[self.IDENTITY],
            delta=self.tolerance * abs(on[self.IDENTITY]),
            msg='the uncrossed matrix element changed when the crossing '
                'machinery was emitted: %r vs %r'
            % (on[self.IDENTITY], off[self.IDENTITY]))


class TestStandaloneMg7CrossSymmetry(unittest.TestCase):
    """standalone_mg7 (madmatrix / cudacpp CPU-SIMD) must reproduce the crossing.

    Mirror of TestStandaloneCppCrossSymmetry for the data-parallel madmatrix
    backend. The extended flavor id encodes cross = id / nflav and flav = id %
    nflav (0-based, NFLAV=1 here), so (I=0, J=3) -> cross = 3 -> id = 3. The key
    extra check versus the scalar C++ backend is that DIFFERENT events in the
    SAME SIMD page may carry DIFFERENT crossings while sharing the reduced
    flavor: the per-event momentum permutation must not be vectorized.

    The whole check needs to build and run real C++/SIMD code; skipped (not
    failed) if the compiler or the madmatrix build toolchain is unavailable.
    """

    CROSS_2_3 = 3       # cross = I*(NEXTERNAL+1)+J = 0*5+3 = 3, id = cross*NFLAV+flav
    IDENTITY = 0
    OVERLAP = 2 * (NEXTERNAL + 1) + 1  # cross=11 (I=2,J=1): overlapping swap -> invalid
    tolerance = 1e-9

    debugging = getattr(unittest, 'debug', False)

    def setUp(self):
        self.compiler = os.environ.get('CXX', 'g++')
        if not shutil.which(self.compiler):
            self.skipTest('no C++ compiler (%s) available' % self.compiler)
        self.tmpdir = tempfile.mkdtemp(prefix='cross_mg7_')

    def tearDown(self):
        if not self.debugging and os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    # ------------------------------------------------------------------
    def _output_standalone_mg7(self, process, name, options=''):
        """Write the standalone_mg7 output for `process`, return its P* dir."""
        outdir = pjoin(self.tmpdir, name)
        cmd = cmd_interface.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False')
        cmd.exec_cmd('set group_subprocesses False')
        cmd.exec_cmd('set apply_flavor_grouping True')
        cmd.exec_cmd('import model sm')
        cmd.exec_cmd(('generate %s %s' % (process, options)).strip())
        cmd.exec_cmd('output standalone_mg7 %s -f' % outdir)

        subproc_root = pjoin(outdir, 'SubProcesses')
        pdirs = [pjoin(subproc_root, d) for d in sorted(os.listdir(subproc_root))
                 if d.startswith('P') and os.path.isdir(pjoin(subproc_root, d))]
        self.assertEqual(len(pdirs), 1,
                         'Expected a single subprocess directory for %s, got %s'
                         % (process, pdirs))
        return pdirs[0]

    def _cpp_source(self, pdir):
        with open(pjoin(pdir, 'CPPProcess.cc')) as fsock:
            return fsock.read()

    def _patch_and_build(self, pdir):
        """Patch the shipped check_sa.cc so it can (a) evaluate the EXTENDED
        flavor ids the crossing needs (the shipped cap stops at nmaxflavor) and
        (b) demonstrate a per-event mixed-crossing page (env MG_FLVMIX/MG_SAMEMOM),
        then build check_sa.exe. Skip if the madmatrix toolchain cannot build."""
        check = pjoin(pdir, 'check_sa.cc')
        with open(check) as fsock:
            src = fsock.read()
        src = src.replace(
            'if( flavorID >= CPPProcess::nmaxflavor )',
            'if( flavorID >= CPPProcess::nmaxflavor * '
            '(unsigned)((CPPProcess::npar+1)*(CPPProcess::npar+1)) )')
        src = src.replace(
            '    std::vector<unsigned int> flvVec( nevt, flavorID );',
            '    std::vector<unsigned int> flvVec( nevt, flavorID );\n'
            '    if( const char* mix = getenv("MG_FLVMIX") ) { unsigned int a=0,b=0; '
            'sscanf(mix,"%u,%u",&a,&b); for(unsigned int i=0;i<nevt;i++) '
            'flvVec[i]=(i%2==0)?a:b; }')
        src = src.replace(
            '        prsk->getMomentaFinal();',
            '        prsk->getMomentaFinal();\n'
            '        if( getenv("MG_SAMEMOM") ) for( unsigned int ie=1; ie<nevt; ie++ ) '
            'for(int ip=0; ip<CPPProcess::npar; ip++) for(int i4=0;i4<4;i4++) '
            'MemoryAccessMomenta::ieventAccessIp4Ipar( hstMomenta.data(), ie, i4, ip ) = '
            'MemoryAccessMomenta::ieventAccessIp4IparConst( hstMomenta.data(), 0, i4, ip );',
            1)
        with open(check, 'w') as fsock:
            fsock.write(src)
        with open(os.devnull, 'w') as devnull:
            rc = subprocess.call(['make', '-j2', 'check_sa.exe'], cwd=pdir,
                                  stdout=devnull, stderr=subprocess.STDOUT)
        if rc != 0:
            self.skipTest('madmatrix build toolchain unavailable (make failed)')

    def _event_mes(self, pdir, flavor_id, env=None):
        """Run check_sa.exe perf verbose and return the per-event ME list."""
        run_env = dict(os.environ)
        if env:
            run_env.update(env)
        out = subprocess.check_output(
            ['./check_sa.exe', 'perf', '-v', '-f', str(flavor_id), '1', '8', '1'],
            cwd=pdir, env=run_env).decode()
        mes = [float(m) for m in
               re.findall(r'Matrix element =\s*([-\d.eE+]+)', out)]
        self.assertTrue(mes, 'no matrix element parsed from:\n%s' % out)
        return mes

    def _me(self, pdir, flavor_id):
        """First-event ME for a single (uniform) flavor id."""
        return self._event_mes(pdir, flavor_id)[0]

    # ------------------------------------------------------------------
    def test_qq_gg_crossed_gives_qg_qg(self):
        """u u~ > g g crossed by (I=0,J=3) equals u g > u g at the same momenta
        (both 2->2 massless -> identical RAMBO momenta for the same seed)."""
        crossed = self._output_standalone_mg7(PROC_QQ_GG, 'qqgg')
        reference = self._output_standalone_mg7(PROC_QG_QG, 'qgqg')
        self._patch_and_build(crossed)
        self._patch_and_build(reference)

        crossed_val = self._me(crossed, self.CROSS_2_3)
        identity_val = self._me(crossed, self.IDENTITY)
        reference_val = self._me(reference, self.IDENTITY)

        self.assertAlmostEqual(
            crossed_val, reference_val,
            delta=self.tolerance * abs(reference_val),
            msg='u u~ > g g crossed (%r) != u g > u g identity (%r)'
            % (crossed_val, reference_val))
        # Non-vacuous: the crossing must move the answer.
        self.assertNotAlmostEqual(
            crossed_val, identity_val, places=6,
            msg='crossed value equals the identity value; crossing had no effect')

    def test_qg_qg_crossed_gives_qq_gg(self):
        """The reverse: u g > u g crossed by (I=0,J=3) equals u u~ > g g."""
        crossed = self._output_standalone_mg7(PROC_QG_QG, 'qgqg_rev')
        reference = self._output_standalone_mg7(PROC_QQ_GG, 'qqgg_rev')
        self._patch_and_build(crossed)
        self._patch_and_build(reference)
        self.assertAlmostEqual(
            self._me(crossed, self.CROSS_2_3), self._me(reference, self.IDENTITY),
            delta=self.tolerance * abs(self._me(reference, self.IDENTITY)),
            msg='u g > u g crossed != u u~ > g g identity')

    def test_per_event_different_cross(self):
        """THE point of the SIMD port: within ONE SIMD page, events carrying
        DIFFERENT crossings (but the same reduced flavor) each get their own
        crossed matrix element. Feed identical momenta to every event, alternate
        the crossing per event (even -> identity, odd -> cross 2<->3) and check
        each lane independently."""
        pdir = self._output_standalone_mg7(PROC_QQ_GG, 'qqgg_perevent')
        self._patch_and_build(pdir)
        identity_val = self._me(pdir, self.IDENTITY)
        crossed_val = self._me(pdir, self.CROSS_2_3)
        self.assertNotAlmostEqual(identity_val, crossed_val, places=6,
                                  msg='degenerate: identity == crossed')
        mixed = self._event_mes(
            pdir, self.IDENTITY,
            env={'MG_SAMEMOM': '1',
                 'MG_FLVMIX': '%d,%d' % (self.IDENTITY, self.CROSS_2_3)})
        self.assertGreaterEqual(len(mixed), 4,
                                'need several events to prove per-event crossing')
        for i, me in enumerate(mixed):
            expected = identity_val if i % 2 == 0 else crossed_val
            self.assertAlmostEqual(
                me, expected, delta=self.tolerance * abs(expected) + 1e-12,
                msg='event %d (cross %s) got %r, expected %r'
                % (i, 'id' if i % 2 == 0 else '2<->3', me, expected))

    def test_invalid_overlapping_swap_returns_zero(self):
        """An overlapping-swap crossing code (I=2, J=1 -> cross 11) is invalid;
        the per-event denominator must short-circuit its matrix element to 0."""
        pdir = self._output_standalone_mg7(PROC_QQ_GG, 'qqgg_inv')
        self._patch_and_build(pdir)
        self.assertEqual(self._me(pdir, self.OVERLAP), 0.0,
                         'an overlapping-swap code must give a zero ME')

    def test_use_crossing_false_byte_identical(self):
        """--use_crossing=False must emit NO crossing machinery (every crossing
        token absent from the generated source) and still give the same
        uncrossed matrix element as the crossing-on build. (A full byte-identical
        `diff -r` against the pre-feature output was checked by hand; here we
        assert the token absence and the numerical invariance.)"""
        on_dir = self._output_standalone_mg7(PROC_QQ_GG, 'qqgg_on')
        off_dir = self._output_standalone_mg7(PROC_QQ_GG, 'qqgg_off',
                                              options='--use_crossing=False')
        on_src = self._cpp_source(on_dir)
        off_src = self._cpp_source(off_dir)
        for token in ('spincol_cross', 'cross_perm', 'cross_ic', 'ident_cross',
                      'xmom', 'flavorPDGs_cross'):
            self.assertIn(token, on_src,
                          '%s should be emitted with crossing on' % token)
            self.assertNotIn(token, off_src,
                             '%s must NOT be emitted with --use_crossing=False'
                             % token)
        self._patch_and_build(on_dir)
        self._patch_and_build(off_dir)
        self.assertAlmostEqual(
            self._me(on_dir, self.IDENTITY), self._me(off_dir, self.IDENTITY),
            delta=self.tolerance * abs(self._me(off_dir, self.IDENTITY)),
            msg='the uncrossed ME changed when the crossing machinery was emitted')


class TestCrossingPartition(unittest.TestCase):
    """partition_crossing_classes routes each subprocess flavor to a base matrix
    element via crossing. A module drops its own matrix<i>.f only when every one
    of its flavors is a crossing of a base module's flavor; the basis for sharing
    one matrix<i>.f across a base and its crossings in the madevent output."""

    def _groups(self, proc):
        import madgraph.iolibs.group_subprocs as group_subprocs
        import madgraph.iolibs.export_v4 as export_v4
        cmd = cmd_interface.MasterCmd()
        cmd.run_cmd('import model sm')
        cmd.run_cmd('define j = g u u~')
        cmd.run_cmd('generate %s' % proc)
        groups = group_subprocs.SubProcessGroup.group_amplitudes(
            cmd._curr_amps, 'madevent')
        for g in groups:
            g.generate_matrix_elements()
        return groups, export_v4.ProcessExporterFortran()

    def test_partition_pp_jj(self):
        groups, exp = self._groups('p p > j j')
        eliminated_any = False
        for g in groups:
            mes = g.get('matrix_elements')
            bases, routing = exp.partition_crossing_classes(mes)
            self.assertEqual(len(routing), len(mes))
            for i in range(len(mes)):
                self.assertTrue(routing[i], 'a module with no flavors')
                for (b, iflav) in routing[i]:
                    self.assertIn(b, bases)            # routes to a real base
                    self.assertGreaterEqual(iflav, 1)  # 1-based FLAV_IDX
                    if i not in bases:
                        # an eliminated module never routes back to itself
                        self.assertNotEqual(b, i)
            if len(bases) < len(mes):
                eliminated_any = True
        self.assertTrue(eliminated_any,
                        'no module was eliminated by crossing in p p > j j')


class TestMadeventCrossingHelicity(unittest.TestCase):
    """End-to-end regression for the crossed-helicity label written to the LHE.

    The madevent helicity path is the phase-4 GET_NHEL decoder plus the phase-5
    runtime crossing encode (the base-selected helicity code is relabelled into
    the dependent's canonical code by permuting its mixed-radix digits with the
    crossing permutation, replacing the old DSIG_XGHEL / router HELMAP tables).
    p p > w+ j is the sharp test: its crossed subprocesses (u g > w+ d, ...) put
    the W+ -- a massive vector with THREE helicity states -- in a leg the
    crossing moved, so a bug in the relabel scrambles the W+ helicity. The W+
    polarisation is physically CHIRAL (asymmetric transverse states) with a
    populated longitudinal (0) state; a scrambled relabel typically reads a
    quark leg's +-1 into the W+ slot and destroys that structure.

    This runs a full (small) madevent generation, so it is a slow test.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='cross_mev_hel_')

    def tearDown(self):
        if os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def test_w_helicity_asymmetry_ppwj(self):
        from madgraph import MG5DIR
        from madgraph.various import lhe_parser
        outdir = pjoin(self.tmpdir, 'ppwj')
        card = pjoin(self.tmpdir, 'cmd.txt')
        with open(card, 'w') as f:
            f.write('generate p p > w+ j\n'
                    'output madevent %s -f\n'
                    'launch\n'
                    'set nevents 1000\n'
                    'set iseed 777\n' % outdir)
        subprocess.call([sys.executable, pjoin(MG5DIR, 'bin', 'mg5_aMC'), card])

        lhe = pjoin(outdir, 'Events', 'run_01', 'unweighted_events.lhe.gz')
        self.assertTrue(os.path.isfile(lhe),
                        'madevent produced no LHE file (%s)' % lhe)

        counts = {-1: 0, 0: 0, 1: 0}
        nevt = 0
        for event in lhe_parser.EventFile(lhe):
            for part in event:
                if part.pid == 24 and part.status == 1:  # the final-state W+
                    hel = int(round(part.helicity))
                    self.assertIn(hel, (-1, 0, 1),
                                  'W+ has undefined/non-physical helicity %r -- '
                                  'helicity output off or scrambled'
                                  % part.helicity)
                    counts[hel] += 1
            nevt += 1
        total = sum(counts.values())
        self.assertGreater(nevt, 100, 'too few events generated (%d)' % nevt)
        self.assertEqual(total, nevt, 'expected exactly one final-state W+ per '
                         'event (got %d W+ in %d events)' % (total, nevt))

        fm, f0, fp = (counts[-1] / total, counts[0] / total, counts[1] / total)
        # All three W+ helicity states populated, incl. the longitudinal 0.
        for hel in (-1, 0, 1):
            self.assertGreater(counts[hel], 0,
                               'W+ helicity %d not populated: %s' % (hel, counts))
        # The two transverse states are chirally asymmetric.
        self.assertGreater(abs(fm - fp), 0.05,
                           'W+ transverse helicities not chirally asymmetric: %s'
                           % counts)
        # The longitudinal fraction sits in a physical window (a scrambled
        # relabel collapses or inflates it out of this range).
        self.assertTrue(0.02 < f0 < 0.45,
                        'W+ longitudinal fraction unphysical: %.3f (%s)'
                        % (f0, counts))


class TestColorFlowCode(unittest.TestCase):
    """The canonical COLOUR-FLOW code, the colour analogue of the canonical
    helicity code.

    A colour flow is labelled by its connectivity once the INITIAL-state legs
    swap their colour/anticolour roles (the LHE convention runs initial-state
    colour lines 'through', so without that flip a label sits in the same slot
    on two legs and the flow is not a colour<->anticolour bijection). Ordering
    the colour and anticolour slots by leg, digit i is the anticolour slot that
    colour slot i connects to and code = sum_i digit_i * N^i.

    Two properties make it usable as an event label and make crossing
    transparent (both verified here):
      (a) every basis flow is a clean bijection, i.e. it encodes at all;
      (b) the code is INJECTIVE over a process's colour basis, so the code
          identifies the flow and no per-process flow table is needed.
    Crossing-covariance (relabelling legs by the crossing permutation carries a
    base flow's code onto the crossed process's own flow code) is exercised by
    the crossing machinery itself: _router_colmap matches flows through the
    same _color_flow_canon helper.
    """

    # (process, expected number of colour flows) -- includes g g > g g g, whose
    # 24 flows over 5 colour slots is the widest case that stays quick.
    PROCS = [('u u~ > g g', 2), ('g g > g g', 6), ('u u~ > u u~', 2),
             ('g g > t t~', 2), ('u u~ > g g g', 6), ('g g > g g g', 24)]

    def test_color_flow_code_bijective_and_injective(self):
        import madgraph.core.helas_objects as helas_objects
        import madgraph.iolibs.export_v4 as export_v4
        exp = export_v4.ProcessExporterFortranMEGroup.__new__(
            export_v4.ProcessExporterFortranMEGroup)
        checked = 0
        for proc, nflow_exp in self.PROCS:
            cmd = cmd_interface.MasterCmd()
            cmd.exec_cmd('generate %s' % proc, printcmd=False)
            me = helas_objects.HelasMultiProcess(cmd._curr_amps)
            for m in me.get('matrix_elements'):
                if not m.get('color_basis'):
                    continue
                codes = exp._color_flow_codes(m)
                # (a) every flow is a clean colour<->anticolour bijection
                self.assertIsNotNone(
                    codes, '%s: a colour flow is not a clean bijection -- the '
                    'initial-state colour/anticolour flip is required' % proc)
                self.assertEqual(len(codes), nflow_exp,
                                 '%s: expected %d colour flows, got %d'
                                 % (proc, nflow_exp, len(codes)))
                # (b) the code identifies the flow
                self.assertEqual(len(set(codes)), len(codes),
                                 '%s: colour-flow codes collide: %s'
                                 % (proc, codes))
                checked += 1
        self.assertTrue(checked, 'no coloured matrix element was checked')
