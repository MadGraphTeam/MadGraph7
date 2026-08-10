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

import itertools
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
# The crossed partner that `g g > q q~` reaches with cross 3 (slot 1 <-> slot 2),
# used by the madmatrix tests that need the crossing to be a RECORDED one.
PROC_GQX_GQX = 'g u~ > g u~'

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


def _massless_2to2(energy, cos_theta):
    """A massless 2->2 point: (leg1_in, leg2_in, leg3_out, leg4_out)."""
    halfe = 0.5 * energy
    sin_theta = math.sqrt(1.0 - cos_theta ** 2)
    return [(halfe, 0.0, 0.0, halfe),
            (halfe, 0.0, 0.0, -halfe),
            (halfe, halfe * sin_theta, 0.0, halfe * cos_theta),
            (halfe, -halfe * sin_theta, 0.0, -halfe * cos_theta)]


# The C-parity de-duplication halves the helicity sum by pairing every row with
# its fully flipped partner. Two all-massless 2->2 processes bracket the rule:
#   u u~ > g g    pure QCD, parity conserving -- every pair matches, so the
#                 reuse ENGAGES and its halve-and-double arithmetic must leave
#                 the answer alone.
#   d u~ > e- ve~ pure charged current, maximally parity violating (V-A) -- only
#                 left-handed fermions couple, so the flipped partner of the one
#                 surviving row is identically zero and the all-or-nothing rule
#                 must REFUSE the reuse for the whole flavor.
PROC_CPARITY_PAIRED = 'u u~ > g g'
PROC_CPARITY_BROKEN = 'd u~ > e- ve~'


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
        return _massless_2to2(self.energy, cos_theta)

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

    # ------------------------------------------------------------------
    # decay chains: the crossing acts at the production level, the whole
    # decay block riding along on its production leg
    # ------------------------------------------------------------------
    def _read_masses(self, pdir):
        """Signed-PDG -> mass, read from the process' generated param_card."""
        card = pjoin(pdir, os.pardir, os.pardir, 'Cards', 'param_card.dat')
        masses = {}
        in_mass = False
        with open(card) as fsock:
            for line in fsock:
                low = line.lower().strip()
                if low.startswith('block mass'):
                    in_mass = True
                    continue
                if in_mass and low.startswith('block'):
                    in_mass = False
                if in_mass:
                    fields = line.split('#')[0].split()
                    if len(fields) == 2:
                        try:
                            masses[int(fields[0])] = float(fields[1])
                        except ValueError:
                            pass
        return masses

    def _massive_2ton(self, pdir, pdgs, seed=7):
        """A phase-space point for a 2->(len(pdgs)-2) process with the leaf
        masses of `pdgs` (signed PDGs, initial two first).

        A decay chain's matrix element is not on any resonance pole at a rambo
        point, so the propagators are finite and the crossed / reference values
        can be compared directly; only the external masses have to be right.
        """
        import madgraph.various.rambo as rambo
        import random
        random.seed(seed)
        masses = self._read_masses(pdir)
        finals = pdgs[2:]
        fmass = rambo.FortranList(len(finals))
        for i, pdg in enumerate(finals):
            fmass[i + 1] = abs(masses.get(abs(pdg), 0.0))
        p_rambo, _ = rambo.RAMBO(len(finals), self.energy, fmass)
        momenta = [(0.5 * self.energy, 0.0, 0.0, 0.5 * self.energy),
                   (0.5 * self.energy, 0.0, 0.0, -0.5 * self.energy)]
        for i in range(1, len(finals) + 1):
            momenta.append((p_rambo[(4, i)], p_rambo[(1, i)],
                            p_rambo[(2, i)], p_rambo[(3, i)]))
        return momenta

    def _assert_decay_crossing(self, base_dir, base_line, ref_line, cross, pdgs):
        """The base decay-chain SMATRIX at a crossing must reproduce a
        fully-generated (--use_crossing=False) build of the crossed decay chain.

        `pdgs` is the crossed leaf signature (from compute_crossing_pdg_entries,
        the order the momenta must be supplied in); it is both the reference
        process order and the momentum order fed to both builds. The base carries
        the crossing through the extended IFLAV, the reference evaluates it as its
        own identity -- the two must agree to machine precision.
        """
        ref_dir = self._generate(ref_line, 'Proc_dc_ref_%d' % cross,
                                 options='--use_crossing=False')
        nflav = self._read_nflav(base_dir)
        momenta = self._massive_2ton(ref_dir, pdgs)
        crossed = self._run(base_dir, momenta, _iflav(cross, 1, nflav=nflav))
        reference = self._run(ref_dir, momenta, IFLAV_IDENTITY)
        self.assertNotEqual(reference, 0.0,
                            'Sanity check failed: %s gives a null matrix element'
                            % ref_line)
        scale = max(abs(crossed), abs(reference), 1e-99)
        self.assertLessEqual(
            abs(crossed - reference) / scale, self.tolerance,
            '%s crossed (cross=%d) disagrees with %s: crossed=%r reference=%r'
            % (base_line, cross, ref_line, crossed, reference))

    def test_decay_chain_crossing_ttbar_jet(self):
        """g u > t t~ u, t > b w+ must reproduce its production crossings.

        The crossing permutes the light partons (a jet moving between the initial
        and the final state); the t decay block (b w+) rides along on the top and
        is never split, and the t~/jet legs move as whole single legs. The base's
        crossing-aware SMATRIX at the crossed flavor index must equal a fully
        generated build of each crossed decay chain.
        """
        base_line = 'g u > t t~ u, t > b w+'
        base = self._generate(base_line, 'Proc_dc_base')
        # (cross code, reference line, crossed leaf signature); the base leaves
        # are [g,u,b,w+,t~,u], NEXTERNAL=6 so CROSS = I*7 + J.
        cases = [
            (6 * 7 + 0, 'u~ u > t t~ g, t > b w+', (-2, 2, 5, 24, -6, 21)),
            (0 * 7 + 6, 'g u~ > t t~ u~, t > b w+', (21, -2, 5, 24, -6, -2)),
        ]
        for cross, ref_line, pdgs in cases:
            with self.subTest(cross=cross):
                self._assert_decay_crossing(base, base_line, ref_line, cross,
                                            pdgs)

    def test_decay_chain_crossing_identical_resonances(self):
        """u u~ > z z g, z > e+ e- exercises the resonance-level denominator.

        Both z decay the same way, so the crossed identical-particle factor is
        NOT a plain count over the crossed leaves (that would double-count the
        two e+/two e-): it is resonance level (the two identical z count once).
        The crossing must rebuild that factor -- IDENT_RESONANCE times the
        countable single legs -- so the crossed value matches a full build.
        """
        base_line = 'u u~ > z z g, z > e+ e-'
        base = self._generate(base_line, 'Proc_dc_zz_base')
        # base leaves [u,u~,e+,e-,e+,e-,g], NEXTERNAL=7 so CROSS = I*8 + J.
        cases = [
            (0 * 8 + 7, 'u g > z z u, z > e+ e-', (2, 21, -11, 11, -11, 11, 2)),
        ]
        for cross, ref_line, pdgs in cases:
            with self.subTest(cross=cross):
                self._assert_decay_crossing(base, base_line, ref_line, cross,
                                            pdgs)


class TestGoodHelCParityDedup(unittest.TestCase):
    """The C-parity de-duplication of the helicity sum must be transparent.

    SMATRIX pairs every helicity row IHEL with FLIP(IHEL), the row with every
    helicity negated. For the first 20 unpolarized calls it evaluates both and
    compares |M|^2 (the scan phase); from then on -- and ONLY if every pair
    matched -- it evaluates the lower-index row once, counts it twice and skips
    its partner, halving the loop (the fast phase).

    Both halves of that contract are checked directly rather than through a
    golden number:

      (a) the premise, per row: for a parity-conserving process the paired rows
          really do have the same |M|^2 at the same momenta, and for a
          parity-violating one they do not. Probed row by row through
          SMATRIXHEL, whose helicity CODE comes from the process' own
          ENCODE_HEL, so this also pins the pairing to the canonical encoding
          rather than to a row index the test guessed.

      (b) the consequence: the plain unpolarized sum is the same before and
          after the fast phase switches on -- both where the reuse engages (the
          halve-and-double arithmetic) and where it must refuse itself. The
          second is the regression: the verdict used to default to "de-duplicate"
          and the validating scan could be skipped entirely (read_good_hel forces
          NTRY past MAXTRIES), so a flavor whose pairs nothing had verified
          silently summed half of its helicities.

    Verified by instrumenting SMATRIX to print DEDUP while writing these: over 30
    successive calls u u~ > g g ends with CSYM true and the fast phase ON from
    call 20, while d u~ > e- ve~ ends with CSYM false and never enters it. The
    two processes really do cover the engage and the refuse branch, so neither
    stability check passes merely because nothing ever happened.
    """

    energy = 1000.0
    cos_theta = 0.3
    # > 20 unpolarized calls, so the last ones are in the fast phase.
    nrepeat = 30
    # The fast phase accumulates 2*|M|^2 at the representative instead of adding
    # the partner separately, so the sum is reassociated: equal to the last bit
    # is not guaranteed, agreement to ~1e-12 is.
    tolerance = 1e-12

    debugging = getattr(unittest, 'debug', False)

    def setUp(self):
        self.cmd = cmd_interface.MasterCmd()
        self.cmd.no_notification()
        self.tmpdir = tempfile.mkdtemp(
            prefix='cparity_debug_' if self.debugging else 'cparity_')

    def tearDown(self):
        if not self.debugging and os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    # ------------------------------------------------------------------
    def _generate(self, process, name):
        """Standalone-output `process`, build the C-parity driver, return its
        P* dir."""
        outdir = pjoin(self.tmpdir, name)
        self.cmd.exec_cmd('set automatic_html_opening False')
        self.cmd.exec_cmd('set group_subprocesses False')
        self.cmd.exec_cmd('set apply_flavor_grouping True')
        self.cmd.exec_cmd('import model sm')
        self.cmd.exec_cmd('generate %s' % process)
        self.cmd.exec_cmd('output standalone %s -f' % outdir)

        subproc_root = pjoin(outdir, 'SubProcesses')
        pdirs = [pjoin(subproc_root, entry)
                 for entry in sorted(os.listdir(subproc_root))
                 if entry.startswith('P')
                 and os.path.isdir(pjoin(subproc_root, entry))]
        self.assertEqual(len(pdirs), 1,
                         'Expected a single subprocess directory for %s, got %s'
                         % (process, pdirs))
        pdir = pdirs[0]
        source = open(pjoin(pdir, 'matrix.f')).read()
        # The probe drives flavor 1 directly, so the process must not have been
        # merged into a multi-flavor matrix element behind our back.
        nflav = re.search(r'PARAMETER\s*\(NFLAV=(\d+)\)', source)
        self.assertTrue(nflav, 'Could not read NFLAV from %s' % pdir)
        self.assertEqual(int(nflav.group(1)), 1,
                         '%s came out with NFLAV=%s; the probe assumes a single '
                         'flavor' % (process, nflav.group(1)))
        ncomb = re.search(r'PARAMETER\s*\(\s*NCOMB=(\d+)\)', source)
        self.assertTrue(ncomb, 'Could not read NCOMB from %s' % pdir)
        self._write_driver(pdir, int(ncomb.group(1)))
        retcode = self._call(['make', 'check'], pdir)
        self.assertEqual(retcode, 0, 'Failed to compile the driver in %s' % pdir)
        return pdir

    @staticmethod
    def _call(command, cwd):
        if logger.isEnabledFor(logging.INFO):
            return subprocess.call(command, cwd=cwd)
        with open(os.devnull, 'w') as devnull:
            return subprocess.call(command, stdout=devnull, stderr=devnull,
                                   cwd=cwd)

    def _write_driver(self, pdir, ncomb):
        """Replace check_sa.f by a driver with the two probes this needs.

        MODE 1 walks the helicity table and reports (|M(h)|^2, |M(-h)|^2) for
        every row, going through ENCODE_HEL so the codes are the process' own.
        MODE 2 calls the plain unpolarized SMATRIX repeatedly at one point, so
        the scan phase and the fast phase can be compared within a single run --
        the de-duplication state lives in SMATRIX and does not survive the
        process.
        """
        driver = '''      PROGRAM CPARITY_DRIVER
      use model_object
      IMPLICIT NONE
      INCLUDE "coupl.inc"
      INCLUDE "nexternal.inc"
      INTEGER NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      REAL*8 P(0:3,NEXTERNAL), ANS, ANSFLIP
      INTEGER I, J, MODE, NREP, IHEL, CODE, FCODE, IDEN_STAR
      INTEGER NHEL_STAR(NEXTERNAL,NCOMB)
      INTEGER THIS(NEXTERNAL), FLIPPED(NEXTERNAL)
      call setpara('param_card.dat')
      OPEN(UNIT=42,FILE='cparity_input.dat',STATUS='OLD')
      READ(42,*) MODE
      DO I=1,NEXTERNAL
         READ(42,*) (P(J,I),J=0,3)
      ENDDO
      IF (MODE.EQ.1) THEN
C        Per-row C-parity probe. SMATRIXHEL selects a single row by its
C        canonical code and undoes the helicity average, the same on both
C        rows of a pair, so the two values are directly comparable.
         CALL GET_NHEL(IDEN_STAR,NHEL_STAR)
         DO IHEL=1,NCOMB
            DO J=1,NEXTERNAL
               THIS(J) = NHEL_STAR(J,IHEL)
               FLIPPED(J) = -NHEL_STAR(J,IHEL)
            ENDDO
            CALL ENCODE_HEL(THIS, CODE)
            CALL ENCODE_HEL(FLIPPED, FCODE)
            CALL SMATRIXHEL(P, CODE, 1, ANS)
            CALL SMATRIXHEL(P, FCODE, 1, ANSFLIP)
            WRITE(*,'(A,3(1X,I6),2(1X,ES25.17))')
     &        'PAIR=', IHEL, CODE, FCODE, ANS, ANSFLIP
         ENDDO
      ELSE
C        The plain unpolarized sum, repeatedly: NTRY_CSYM crosses its
C        threshold part way through and the fast phase takes over.
         READ(42,*) NREP
         DO I=1,NREP
            CALL SMATRIX(P,1,ANS)
            WRITE(*,'(A,1X,I6,1X,ES25.17)') 'ANS=', I, ANS
         ENDDO
      ENDIF
      CLOSE(42)
      END
'''
        with open(pjoin(pdir, 'check_sa.f'), 'w') as fsock:
            fsock.write(driver % {'ncomb': ncomb})

    def _probe(self, pdir, lines):
        with open(pjoin(pdir, 'cparity_input.dat'), 'w') as fsock:
            fsock.write('\n'.join(lines) + '\n')
        return subprocess.Popen(['./check'], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                cwd=pdir).communicate()[0].decode()

    def _momentum_lines(self):
        return [' '.join('%.17e' % component for component in mom)
                for mom in _massless_2to2(self.energy, self.cos_theta)]

    def _pairs(self, pdir):
        """[(row, |M(h)|^2, |M(-h)|^2)] over the whole helicity table."""
        output = self._probe(pdir, ['1'] + self._momentum_lines())
        pairs = [(int(row), float(direct), float(flipped))
                 for row, _code, _fcode, direct, flipped
                 in re.findall(r'PAIR=\s+(\d+)\s+(\d+)\s+(\d+)\s+'
                               r'(\S+)\s+(\S+)', output)]
        self.assertTrue(pairs, 'no C-parity pair read from %s, got:\n%s'
                        % (pdir, output))
        return pairs

    def _repeated_sums(self, pdir):
        """The unpolarized SMATRIX value of each of `nrepeat` successive calls."""
        output = self._probe(pdir, ['2'] + self._momentum_lines()
                             + ['%d' % self.nrepeat])
        values = [float(value)
                  for _call, value in re.findall(r'ANS=\s+(\d+)\s+(\S+)', output)]
        self.assertEqual(len(values), self.nrepeat,
                         'expected %d matrix elements from %s, got %d:\n%s'
                         % (self.nrepeat, pdir, len(values), output))
        return values

    def _assert_sum_is_stable(self, pdir, label):
        """Every repeated call must give the first call's value.

        Call 1 is in the scan phase (full helicity sum, both members of every
        pair evaluated); the last calls are past the threshold. If the reuse is
        wrong -- a missing factor of two, or a de-duplication applied to a
        flavor whose pairs do not match -- the value steps part way through.
        """
        values = self._repeated_sums(pdir)
        reference = values[0]
        self.assertNotEqual(reference, 0.0,
                            '%s gives a null matrix element' % label)
        for index, value in enumerate(values, start=1):
            self.assertLessEqual(
                abs(value - reference), self.tolerance * abs(reference),
                '%s: call %d gives %r but call 1 gave %r -- the C-parity '
                'de-duplication changed the unpolarized sum'
                % (label, index, value, reference))

    # ------------------------------------------------------------------
    def test_cparity_pairs_match_for_qcd(self):
        """Parity-conserving: every row equals its fully flipped partner.

        This is the premise the fast phase rests on. Checked row by row, so a
        pairing built on the wrong encoding fails here rather than silently
        halving the sum somewhere else.
        """
        pdir = self._generate(PROC_CPARITY_PAIRED, 'Proc_cparity_qcd')
        pairs = self._pairs(pdir)
        nonzero = 0
        for row, direct, flipped in pairs:
            scale = max(abs(direct), abs(flipped))
            if scale == 0.0:
                continue
            nonzero += 1
            self.assertLessEqual(
                abs(direct - flipped), 1e-10 * scale,
                '%s row %d: |M(h)|^2=%r but |M(-h)|^2=%r; the C-parity pairing '
                'the de-duplication relies on does not hold'
                % (PROC_CPARITY_PAIRED, row, direct, flipped))
        self.assertGreater(nonzero, 1,
                           'only %d non-zero helicity row(s) in %s: the pairing '
                           'is not being exercised'
                           % (nonzero, PROC_CPARITY_PAIRED))

    def test_cparity_pairs_broken_for_charged_current(self):
        """Maximally parity-violating: at least one pair must NOT match.

        Without this the "all-or-nothing refusal" half of the rule would never
        be exercised -- if every process in the suite happened to be
        parity-conserving, a de-duplication that never refuses would pass.
        """
        pdir = self._generate(PROC_CPARITY_BROKEN, 'Proc_cparity_cc')
        pairs = self._pairs(pdir)
        mismatched = [(row, direct, flipped)
                      for row, direct, flipped in pairs
                      if abs(direct - flipped)
                      > 1e-10 * max(abs(direct), abs(flipped), 1e-99)]
        self.assertTrue(
            mismatched,
            '%s: every helicity row matched its flipped partner, so this '
            'process does not test the refusal path any more' % PROC_CPARITY_BROKEN)

    def test_dedup_leaves_the_paired_sum_unchanged(self):
        """The reuse engages here, and must not move the answer."""
        pdir = self._generate(PROC_CPARITY_PAIRED, 'Proc_cparity_qcd_sum')
        self._assert_sum_is_stable(pdir, PROC_CPARITY_PAIRED)

    def test_refused_dedup_leaves_the_broken_sum_unchanged(self):
        """The regression: the reuse must refuse itself here.

        If it does not, the fast phase drops every row whose partner is zero and
        doubles the wrong ones, and the sum moves at call 21.
        """
        pdir = self._generate(PROC_CPARITY_BROKEN, 'Proc_cparity_cc_sum')
        self._assert_sum_is_stable(pdir, PROC_CPARITY_BROKEN)


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
    (grouped) madevent output decode one; an output that cannot must not quietly
    produce a matrix element missing those subprocesses -- it gets the recorded
    crossings expanded back into explicit subprocesses instead, so the result is
    the complete uncrossed output and no user flag is required.
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

    def _output(self, fmt, name, options='', process=PROC_QG_QG, setup=(),
                out_options=''):
        """Run generate+output for `fmt`; returns the output directory.

        `options` goes on the generate line, `out_options` on the output line.
        """
        out = pjoin(self.tmpdir, name)
        cmd = cmd_interface.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False')
        for line in setup:
            cmd.exec_cmd(line)
        cmd.exec_cmd('import model sm')
        cmd.exec_cmd(('generate %s %s' % (process, options)).strip())
        cmd.exec_cmd(('output %s %s -f %s' % (fmt, out, out_options)).strip())
        return out

    @staticmethod
    def _subprocesses(out_dir):
        path = pjoin(out_dir, 'SubProcesses')
        return sorted(name for name in os.listdir(path)
                      if name.startswith('P'))

    def test_unsupported_output_accepts_crossing(self):
        """Crossing on must NOT be refused by an output that cannot read it.

        The crossed subprocesses are recorded as metadata at generation, so an
        output with no crossing machinery gets them expanded back into explicit
        subprocesses instead of erroring out. Refusing here used to force the
        user to pass --use_crossing=False even for a process that folds no
        crossing at all (u g > u g folds none), which is why the gate moved from
        the flag to the data.
        """
        for fmt in self.UNSUPPORTED_FORMATS:
            with self.subTest(format=fmt):
                with_crossing = self._output(fmt, 'on_%s' % fmt)
                without = self._output(fmt, 'off_%s' % fmt,
                                       options='--use_crossing=False')
                self.assertEqual(self._subprocesses(with_crossing),
                                 self._subprocesses(without),
                                 '%s output differs with crossing on' % fmt)

    def test_ungrouped_madevent_expands_folded_crossings(self):
        """A folding process must lose nothing on an output without crossing.

        p p > j j QCD=0 really does fold crossings, so this is the case where a
        silently-missing subprocess would change the cross-section: the
        ungrouped madevent output (no crossing machinery) must come out with the
        very same subprocesses as an explicitly uncrossed generation.
        """
        ungrouped = ('set group_subprocesses False',)
        on = self._output('madevent', 'me_on', process='p p > j j QCD=0',
                          setup=ungrouped)
        off = self._output('madevent', 'me_off', process='p p > j j QCD=0',
                           options='--use_crossing=False', setup=ungrouped)
        subs_on = self._subprocesses(on)
        self.assertEqual(subs_on, self._subprocesses(off))
        # Guard the guard: a build that collapsed everything into one directory
        # would satisfy the equality above only if both sides were broken.
        self.assertGreater(len(subs_on), 1,
                           'expected several crossed subprocesses, got %s'
                           % subs_on)

    def test_unsupported_output_accepted_without_crossing(self):
        """--use_crossing=False must let the very same output through.

        Without this the test above would be satisfied by an exporter that is
        simply broken, rather than by one gating on the crossing request.
        """
        for fmt in self.UNSUPPORTED_FORMATS:
            with self.subTest(format=fmt):
                self._output(fmt, 'ok_%s' % fmt,
                             options='--use_crossing=False')

    def test_folding_output_expands_when_crossing_turned_off(self):
        """--use_crossing=False on the output line must stay a COMPLETE output.

        The generation folds the crossed subprocesses onto their base and the
        standalone backends reach them through the base's crossing-aware
        SMATRIX/sigmaKin. Dropping that machinery at output time therefore has to
        put the folded subprocesses back, or the output silently loses those
        partonic contributions -- the exact trap the flag is documented never to
        spring. q q > q q (q = u d u~ d~) really does fold: it collapses to one
        directory with crossing on.
        """
        setup = ('define q = u d u~ d~',)
        proc = 'q q > q q'
        for fmt in ('standalone', 'standalone_mg7'):
            with self.subTest(format=fmt):
                on = self._output(fmt, 'fold_on_%s' % fmt, process=proc,
                                  setup=setup)
                gen_off = self._output(fmt, 'fold_gen_%s' % fmt, process=proc,
                                       setup=setup,
                                       options='--use_crossing=False')
                out_off = self._output(fmt, 'fold_out_%s' % fmt, process=proc,
                                       setup=setup,
                                       out_options='--use_crossing=False')
                self.assertEqual(self._subprocesses(gen_off),
                                 self._subprocesses(out_off),
                                 '%s: --use_crossing=False on the output line '
                                 'kept the crossings folded' % fmt)
                # Guard the guard: both sides would agree if nothing ever folded.
                self.assertLess(len(self._subprocesses(on)),
                                len(self._subprocesses(out_off)),
                                '%s: expected %s to fold crossings with the '
                                'crossing on' % (fmt, proc))

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
        for token in ('spincol_cross', 'cross_perm_ic', 'spincol_part',
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
    CROSS_TO_QQ_GG = 23 # the crossing taking g g > q q~ to q q~ > g g
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
    def _output_standalone_mg7(self, process, name, options='',
                               out_options='', color_basis=None):
        """Write the standalone_mg7 output for `process`, return its P* dir.

        `options` goes on the generate line, `out_options` on the output line.

        color_basis is only passed when the caller compares this output against
        another one number-by-number: the colour sum is accumulated in a
        different order in each basis, so mixing bases moves the last few digits
        (~1e-7 relative) and swamps the 1e-9 tolerance."""
        outdir = pjoin(self.tmpdir, name)
        cmd = cmd_interface.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False')
        cmd.exec_cmd('set group_subprocesses False')
        cmd.exec_cmd('set apply_flavor_grouping True')
        if color_basis:
            cmd.exec_cmd('set color_basis %s' % color_basis)
        cmd.exec_cmd('import model sm')
        cmd.exec_cmd(('generate %s %s' % (process, options)).strip())
        cmd.exec_cmd(('output standalone_mg7 %s -f %s'
                      % (outdir, out_options)).strip())

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
        # FPTYPE=d, not the makefile default: the default is 'm' (mixed), whose
        # colour algebra runs in single precision, so two evaluations of the
        # same |M|^2 that accumulate in a different order (a crossed base vs the
        # crossed process computed on its own) part company at ~1e-7 relative --
        # a hundredfold above the 1e-9 tolerance these tests compare at.
        build_env = dict(os.environ, FPTYPE='d')
        with open(os.devnull, 'w') as devnull:
            rc = subprocess.call(['make', '-j2', 'check_sa.exe'], cwd=pdir,
                                  stdout=devnull, stderr=subprocess.STDOUT,
                                  env=build_env)
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

    def _output_folded_gg_qqx(self, name):
        """Write a multiprocess in which `g g > q q~` is the FOLDED base of its
        crossings, and return that P* dir.

        The good-helicity scan only visits the crossings this ME actually
        records (cross_recorded / _scanned_crossings), so a crossed matrix
        element can only be asked for on a base that folded it in. A bare
        `generate u u~ > g g` records nothing, so the crossings below have to
        come from a real multiparticle expansion: `pq pq > pq pq` with
        pq = g u u~ folds `g u~ > g u~` (cross 3) and `u u~ > g g` (cross 23)
        onto the `g g > q q~` base -- the same two directions the standalone
        references below compute on their own.

        The trace basis is forced because the sibling all-gluon dir of this
        multiprocess cannot be written with the DDM default (unrelated to
        crossing: color_flow_decomposition has no single flow per DDM element).
        """
        outdir = pjoin(self.tmpdir, name)
        cmd = cmd_interface.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False')
        cmd.exec_cmd('set group_subprocesses False')
        cmd.exec_cmd('set apply_flavor_grouping True')
        cmd.exec_cmd('set color_basis trace')
        cmd.exec_cmd('import model sm')
        cmd.exec_cmd('define pq = g u u~')
        cmd.exec_cmd('generate pq pq > pq pq')
        cmd.exec_cmd('output standalone_mg7 %s -f' % outdir)

        subproc_root = pjoin(outdir, 'SubProcesses')
        pdirs = [pjoin(subproc_root, d) for d in sorted(os.listdir(subproc_root))
                 if d.startswith('P') and 'gg_QQx' in d
                 and os.path.isdir(pjoin(subproc_root, d))]
        self.assertEqual(len(pdirs), 1,
                         'expected exactly one folded g g > q q~ dir, got %s'
                         % pdirs)
        demo = pjoin(pdirs[0], 'crossing_demo.dat')
        self.assertTrue(os.path.exists(demo),
                        'no crossing was folded onto %s' % pdirs[0])
        with open(demo) as fsock:
            recorded = [int(tok) for tok in fsock.read().split()]
        for wanted in (self.CROSS_2_3, self.CROSS_TO_QQ_GG):
            self.assertIn(wanted, recorded,
                          'crossing %d is not recorded in %s (got %s); the '
                          'good-hel scan would not have scanned it'
                          % (wanted, demo, recorded))
        return pdirs[0]

    # ------------------------------------------------------------------
    def test_gg_qqx_crossed_gives_qg_qg(self):
        """g g > q q~ crossed by (I=0,J=3) equals g u~ > g u~ at the same momenta
        (both 2->2 massless -> identical RAMBO momenta for the same seed).

        The base must be one that FOLDED this crossing in: the good-hel scan
        only visits recorded crossings, so a bare `generate u u~ > g g` (which
        records none) can no longer be driven with an arbitrary crossing code.
        See _output_folded_gg_qqx."""
        crossed = self._output_folded_gg_qqx('ggqqx')
        reference = self._output_standalone_mg7(PROC_GQX_GQX, 'gqxgqx',
                                                color_basis='trace')
        self._patch_and_build(crossed)
        self._patch_and_build(reference)

        crossed_val = self._me(crossed, self.CROSS_2_3)
        identity_val = self._me(crossed, self.IDENTITY)
        reference_val = self._me(reference, self.IDENTITY)

        self.assertAlmostEqual(
            crossed_val, reference_val,
            delta=self.tolerance * abs(reference_val),
            msg='g g > q q~ crossed (%r) != g u~ > g u~ identity (%r)'
            % (crossed_val, reference_val))
        # Non-vacuous: the crossing must move the answer.
        self.assertNotAlmostEqual(
            crossed_val, identity_val, places=6,
            msg='crossed value equals the identity value; crossing had no effect')

    def test_gg_qqx_crossed_gives_qq_gg(self):
        """The other recorded direction: g g > q q~ crossed to u u~ > g g."""
        crossed = self._output_folded_gg_qqx('ggqqx_rev')
        reference = self._output_standalone_mg7(PROC_QQ_GG, 'qqgg_rev',
                                                color_basis='trace')
        self._patch_and_build(crossed)
        self._patch_and_build(reference)
        reference_val = self._me(reference, self.IDENTITY)
        self.assertAlmostEqual(
            self._me(crossed, self.CROSS_TO_QQ_GG), reference_val,
            delta=self.tolerance * abs(reference_val),
            msg='g g > q q~ crossed != u u~ > g g identity')

    def test_per_event_different_cross(self):
        """THE point of the SIMD port: within ONE SIMD page, events carrying
        DIFFERENT crossings (but the same reduced flavor) each get their own
        crossed matrix element. Feed identical momenta to every event, alternate
        the crossing per event (even -> identity, odd -> cross 2<->3) and check
        each lane independently.

        Both codes used here are RECORDED crossings of the folded base, which is
        what the good-hel scan covers (see _output_folded_gg_qqx)."""
        pdir = self._output_folded_gg_qqx('ggqqx_perevent')
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
        for token in ('spincol_cross', 'cross_perm_ic', 'spincol_part',
                      'ident_cross', 'xmom', 'ids_base'):
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

    def test_use_crossing_false_on_the_output_line(self):
        """--use_crossing=False on the OUTPUT line must reach the exporter.

        The flag used to be read by the generate command only, so passing it to
        `output` was silently a no-op: the whole crossing machinery (preamble,
        per-crossing good-helicity tables, NSF-blended external calls, the
        cNGoodMaxCross loop bound) was emitted anyway. Writing the same source
        as the generate-time flag is the sharpest statement of the fix, since
        that build is the one covered by the tests above.
        """
        gen_dir = self._output_standalone_mg7(PROC_QQ_GG, 'qqgg_genoff',
                                              options='--use_crossing=False')
        out_dir = self._output_standalone_mg7(PROC_QQ_GG, 'qqgg_outoff',
                                              out_options='--use_crossing=False')
        out_src = self._cpp_source(out_dir)
        self.assertEqual(self._cpp_source(gen_dir), out_src,
                         '--use_crossing=False writes a different source on the '
                         'output line than on the generate line')
        # Guard the guard: an exporter that never emits the machinery would
        # satisfy the equality above with both sides broken.
        on_src = self._cpp_source(
            self._output_standalone_mg7(PROC_QQ_GG, 'qqgg_defaulton'))
        for token in ('spincol_cross', 'cross_perm_ic', 'ident_cross',
                      'cNGoodMaxCross'):
            self.assertIn(token, on_src,
                          '%s should be emitted with crossing on' % token)
            self.assertNotIn(token, out_src,
                             '%s must NOT survive --use_crossing=False on the '
                             'output line' % token)


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
        # partition_crossing_classes operates on the FULL (unmerged) matrix-element
        # list -- exactly what the madevent output reconstructs from the recorded
        # crossings before grouping. Generate unmerged here so the routing has the
        # crossed modules to eliminate (the default merge_crossing='record' would
        # fold them away at generation, leaving nothing to route).
        old = os.environ.get('MG_MERGE_CROSSING')
        os.environ['MG_MERGE_CROSSING'] = 'off'
        try:
            cmd.run_cmd('generate %s' % proc)
        finally:
            if old is None:
                os.environ.pop('MG_MERGE_CROSSING', None)
            else:
                os.environ['MG_MERGE_CROSSING'] = old
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


class TestCrossingRecycledHelicityUnion(unittest.TestCase):
    """crossgroup_helunion.dat must carry the helicity map the RECYCLED optim can
    actually realise: the crossing's NSF SIGN flips, with NO slot permutation.

    A crossing base's matrix<b>_optim.f is entered by every member of its class,
    so gen_ximprove has to bake it over a helicity set that covers them all. The
    trap is that there are two different base->base helicity maps and only one of
    them applies here. matrix<b>_optim.f bakes its configs into the HELAS calls
    and receives only (PUSE, IC): IC carries the crossing's sign flips, and
    NOTHING carries its slot permutation. So the transform it realises is
    tau[h][k] = base_row[h][k]*SGN[k], and optim row h is non-zero for the
    crossing iff tau[h] is good for the base -- the union to bake is
    G_base U tau(G_base).

    Feeding it the other map instead -- the GHREMAP sigma[h][k] =
    base_row[h][PERM[k]]*SGN[k], which matrix<b>_orig.f does realise because it
    takes NHEL at run time -- looks equally plausible and is silently wrong. It
    cost -28.5% on the q q~ > q q~ cross section (5.19e6 -> 3.71e6 pb): the
    routed t-channel subprocess needs 4 of the base's 16 rows and the sigma union
    supplied 2 of them. Both maps are permutations, both are involutions here,
    and both give a set that is invariant under themselves, so nothing about the
    set's shape gives the mistake away -- hence this test on the map itself.

    Run-free (no integration): it checks the generation-time map directly, on the
    same q q~ > q q~ class whose cross section paid for it.
    """

    PROCESS = 'q q~ > q q~'

    def _class(self, proc):
        """(exporter, base matrix element, cross) for a routed crossing of `proc`
        that moves at least one leg between the initial and the final state."""
        import madgraph.iolibs.group_subprocs as group_subprocs
        import madgraph.iolibs.export_v4 as export_v4
        cmd = cmd_interface.MasterCmd()
        # apply_flavor_grouping False is the setting that puts q q~ > q q~ in ONE
        # group of three matrix elements with a crossing router -- and the one
        # whose cross section the sigma union broke. --no_save keeps it out of the
        # user's configuration.
        cmd.run_cmd('set apply_flavor_grouping False --no_save')
        cmd.run_cmd('import model sm')
        cmd.run_cmd('define q = u d s c')
        cmd.run_cmd('define q~ = u~ d~ s~ c~')
        # As in TestCrossingPartition: route the UNMERGED list, which is what the
        # madevent output reconstructs before grouping.
        old = os.environ.get('MG_MERGE_CROSSING')
        os.environ['MG_MERGE_CROSSING'] = 'off'
        try:
            cmd.run_cmd('generate %s' % proc)
        finally:
            if old is None:
                os.environ.pop('MG_MERGE_CROSSING', None)
            else:
                os.environ['MG_MERGE_CROSSING'] = old
        groups = group_subprocs.SubProcessGroup.group_amplitudes(
            cmd._curr_amps, 'madevent')
        exp = export_v4.ProcessExporterFortranMEGroup()
        out = []
        for g in groups:
            g.generate_matrix_elements()
            mes = g.get('matrix_elements')
            bases, routing = exp.partition_crossing_classes(mes)
            for idep, route in enumerate(routing or []):
                if route is None or idep in bases:
                    continue
                for (base_index, iflav) in route:
                    base_me = mes[base_index]
                    nflav = len(base_me.get_external_flavors_with_iden())
                    out.append((exp, base_me, (iflav - 1) // nflav))
        return out

    def test_helunion_map_is_the_sign_flip_not_the_permutation(self):
        classes = self._class(self.PROCESS)
        self.assertTrue(classes,
                        'no subprocess of %s is routed through a crossing, so '
                        'this test checks nothing' % self.PROCESS)
        differs = 0
        for exp, base_me, cross in classes:
            bh = [tuple(x) for x in base_me.get_helicity_matrix()]
            tables = exp.compute_crossing_tables(base_me)
            nx = tables['nexternal']
            perm = [tables['perm'][cross * nx + k] for k in range(nx)]
            sgn = [tables['ic'][cross * nx + k] for k in range(nx)]

            tau = exp._crossgroup_base_helsignmap(base_me, cross)
            self.assertIsNotNone(
                tau, 'tau is not a permutation for cross %d: the helicity states '
                'of the crossed legs must be closed under negation' % cross)

            # The defining property: tau moves NO helicity between slots. Row
            # tau[h] is row h with the crossed legs' helicity negated in place.
            # Baking sigma instead breaks exactly this.
            for h, row in enumerate(bh, 1):
                self.assertEqual(
                    bh[tau[h - 1] - 1],
                    tuple(row[k] * sgn[k] for k in range(nx)),
                    'crossgroup_helunion row %d of cross %d is not the pure '
                    'sign flip: a recycled optim cannot apply a slot '
                    'permutation' % (h, cross))

            # ... and for a crossing that does move legs across, sigma is a
            # genuinely different map, so getting this wrong is not academic.
            sigma = exp._helicity_row_permutation(
                *exp._crossed_helicity_configs(base_me, cross))
            if perm != list(range(nx)) and sigma is not None and sigma != tau:
                differs += 1
        self.assertTrue(
            differs,
            'sigma and tau coincide for every crossing of %s, so this process '
            'cannot tell the two apart -- pick one that can' % self.PROCESS)


class TestCrossingConfigMap(unittest.TestCase):
    """_crossgroup_configmap must send a crossed subprocess's multi-channel
    CONFIG to the base diagram of the same topology under the crossing.

    The dependent's genps samples its OWN config's poles, but the shared base
    SMATRIX enhances AMP2(channel) in the BASE's diagram numbering, so `channel`
    has to be translated on the way in. Any bijective pairing still sums to the
    right integral -- what a wrong pairing wrecks is the importance sampling:
    each channel's weight ends up on the wrong amplitude, so the variance blows
    up and the error madevent quotes stops meaning anything.

    That failure is invisible from the outside. The function returns the
    IDENTITY when it cannot match the diagrams, which is indistinguishable from
    the common and perfectly legitimate case of a crossing-covariant numbering;
    the matrix elements still agree to every digit, and only the stability of
    the cross section suffers. So it is checked here, on the map itself:
    whatever base diagram a config is routed to must carry the same internal
    propagators as the dependent's own diagram, once the crossing has relabelled
    the legs.

    Both crossing paths call it -- the within-group router (Track A,
    write_matrix_router_file) and the cross-group auto_dsig fill (Track B,
    _dsig_crossgroup_fills) -- so both are covered.
    """

    @staticmethod
    def _canon(sub, allset):
        return min(sub, allset - sub, key=lambda x: (len(x), sorted(x)))

    @classmethod
    def _propagators(cls, me):
        """Per diagram number, its internal propagators as a frozenset of
        (canonical external-leg subset, |PDG|).

        Recomputed here rather than taken from the exporter's own topology
        helper on purpose: this is the reference the map is judged against, so
        it must not move when that helper does. A propagator is pinned down by
        the external legs whose momenta flow through it -- a subset and its
        complement being the same propagator, hence the canonical choice -- plus
        the particle running in it. |PDG| and not PDG, because crossing a leg
        reverses the flow through every propagator on its path and so conjugates
        them; the magnitude is what survives the relabelling.
        """
        nx, nini = me.get_nexternal_ninitial()
        model = me.get('processes')[0].get('model')
        npdg = model.get_first_non_pdg()
        allset = frozenset(range(1, nx + 1))
        out = {}
        for diag in me.get('diagrams'):
            sch, tch = diag.get('amplitudes')[0].get_s_and_t_channels(
                nini, model, npdg)
            ext = {i: frozenset([i]) for i in range(1, nx + 1)}
            props = set()
            for vert in list(sch) + list(tch):
                legs = vert.get('legs')
                daughters = [l.get('number') for l in legs[:-1]]
                sub = frozenset().union(*[ext.get(d, frozenset([d]))
                                          for d in daughters]) if daughters \
                    else frozenset()
                ext[legs[-1].get('number')] = sub
                # the last t-channel 'propagator' is a single external leg
                if len(cls._canon(sub, allset)) >= 2:
                    props.add((cls._canon(sub, allset),
                               abs(legs[-1].get('id'))))
            out[diag.get('number')] = frozenset(props)
        return out, nx

    def _routed_pairs(self, procs, defs=(), unfold=False):
        """Every (track, dep_me, base_me, crossing) a generation routes through
        a shared matrix element, collected from BOTH crossing paths.

        unfold=True sets MG_MERGE_CROSSING=off so the crossed modules are kept
        instead of folded away at generation -- that is what leaves within-group
        (Track A) routers to find. With the default 'record' the same processes
        come back as whole crossed GROUPS and go through Track B instead, so the
        two settings exercise different code and neither subsumes the other.
        """
        import madgraph.iolibs.group_subprocs as group_subprocs
        import madgraph.iolibs.export_v4 as export_v4
        cmd = cmd_interface.MasterCmd()
        cmd.run_cmd('import model sm')
        for definition in defs:
            cmd.run_cmd(definition)
        old = os.environ.get('MG_MERGE_CROSSING')
        if unfold:
            os.environ['MG_MERGE_CROSSING'] = 'off'
        try:
            for i, proc in enumerate(procs):
                cmd.run_cmd('%s %s' % ('generate' if i == 0 else 'add process',
                                       proc))
        finally:
            if unfold:
                if old is None:
                    os.environ.pop('MG_MERGE_CROSSING', None)
                else:
                    os.environ['MG_MERGE_CROSSING'] = old
        groups = group_subprocs.SubProcessGroup.group_amplitudes(
            cmd._curr_amps, 'madevent')
        for group in groups:
            group.generate_matrix_elements()
        exp = export_v4.ProcessExporterFortranMEGroup()
        exp.opt['use_crossing'] = True

        pairs = []

        def add(track, dep, base, iflav):
            nflav_base = len(base.get_external_flavors_with_iden())
            pairs.append((track, dep, base, (iflav - 1) // nflav_base))

        for group in groups:                      # Track A, within-group
            mes = group.get('matrix_elements')
            bases, routing = exp.partition_crossing_classes(mes)
            for i, route in enumerate(routing):
                if i in bases:
                    continue
                for (b, iflav) in route:
                    add('A', mes[i], mes[b], iflav)
        for (gi, mi), cg in exp.compute_crossgroup_routing(groups).items():
            dep = groups[gi].get('matrix_elements')[mi]
            for iflav in cg['flav_idx']:          # Track B, cross-group
                add('B', dep, cg['base_me'], iflav)

        # the flavors of one module usually share a (base, crossing)
        seen, out = set(), []
        for pair in pairs:
            key = (pair[0], id(pair[1]), id(pair[2]), pair[3])
            if key not in seen:
                seen.add(key)
                out.append(pair)
        return exp, out

    def _check(self, procs, defs=(), unfold=False, min_pairs=1):
        exp, pairs = self._routed_pairs(procs, defs=defs, unfold=unfold)
        checked = 0
        for (track, dep, base, cross) in pairs:
            ngraphs = len(dep.get('diagrams'))
            if len(base.get('diagrams')) != ngraphs:
                # both call sites leave a mismatched diagram count alone
                continue
            label = 'Track %s: %s <- %s (crossing %d)' % (
                track, dep.get('processes')[0].shell_string(),
                base.get('processes')[0].shell_string(), cross)
            cmap = exp._crossgroup_configmap(dep, base, cross)
            self.assertEqual(
                sorted(cmap), list(range(1, ngraphs + 1)),
                '%s: the config map is not a permutation of the %d diagrams'
                % (label, ngraphs))
            dprops, nx = self._propagators(dep)
            bprops, _ = self._propagators(base)
            perm = exp.get_crossing_permutation(cross, nx)[0]
            d2b = {k + 1: perm[k] + 1 for k in range(nx)}
            allset = frozenset(range(1, nx + 1))
            fmt = lambda ps: sorted((sorted(s), pdg) for (s, pdg) in ps)
            for d in range(1, ngraphs + 1):
                want = frozenset(
                    (self._canon(frozenset(d2b[l] for l in sub), allset), pdg)
                    for (sub, pdg) in dprops[d])
                self.assertEqual(
                    bprops[cmap[d - 1]], want,
                    '%s:\n  config %d is routed to base diagram %d, but that '
                    'diagram is not this one crossed.\n'
                    '  base diagram %d propagators: %s\n'
                    '  dependent diagram %d crossed: %s\n'
                    '  (a silent fallback to the identity map looks exactly '
                    'like this; it costs cross-section stability, not the '
                    'cross section itself)'
                    % (label, d, cmap[d - 1], cmap[d - 1],
                       fmt(bprops[cmap[d - 1]]), d, fmt(want)))
            checked += 1
        self.assertGreaterEqual(
            checked, min_pairs,
            'expected at least %d routed subprocess(es) to check, got %d -- '
            'the generation no longer exercises the crossing router'
            % (min_pairs, checked))

    def test_configmap_cross_group(self):
        """Track B. g g > t t~ u u~ and its crossing u u~ > t t~ g g land in two
        separate groups, so the second routes to the first's matrix element
        through the cross-group path. Both have 36 diagrams, two of which share
        a pure leg-subset topology and are told apart only by the particle in
        the propagator: a gluon, versus the auxiliary field that carries the
        four-gluon vertex. Matching on the leg subsets alone collapses those two
        into one signature, the pairing stops being a bijection, and the whole
        map silently degrades to the identity -- which mis-pairs EVERY channel,
        not just the ambiguous two.
        """
        self._check(['g g > t t~ u u~', 'u u~ > t t~ g g'])

    def test_configmap_within_group(self):
        """Track A. The same ambiguity reaches the within-group router: with the
        crossed modules kept rather than folded away, p p > t t~ j j routes
        g Q~ > t t~ g Q~ to g Q > t t~ g Q, again 36 diagrams with the same
        gluon / four-gluon-auxiliary pair among them.
        """
        self._check(['p p > t t~ j j'], defs=['define j = g u u~'],
                    unfold=True)

    def test_configmap_stays_correct_where_it_already_worked(self):
        """Control: p p > j j routes two modules and its diagrams have always
        been matched cleanly. Sharpening the topology signature enough to split
        the ambiguous pair above must not start REJECTING these -- an invariant
        that is not crossing-covariant would fail here.
        """
        self._check(['p p > j j'], defs=['define j = g u u~'], unfold=True,
                    min_pairs=2)

    def test_unmatchable_diagrams_are_reported(self):
        """The fallback must say so. Nothing downstream can detect a degraded
        config map -- it is a legal bijection that merely samples badly -- so the
        one chance to notice is at generation.

        Fed two processes that are not crossings of each other (a synthetic
        stand-in for any pair the topology signature cannot match, since the
        physical pairs are all matched again now), the map must come back as the
        identity AND name both matrix elements.
        """
        import madgraph.core.helas_objects as helas_objects
        import madgraph.iolibs.export_v4 as export_v4
        mes = []
        for proc in ('u u~ > t t~ g g', 'u u~ > t t~ u u~'):
            cmd = cmd_interface.MasterCmd()
            cmd.exec_cmd('import model sm', printcmd=False)
            cmd.exec_cmd('generate %s' % proc, printcmd=False)
            mes.append(helas_objects.HelasMultiProcess(
                cmd._curr_amps).get_matrix_elements()[0])
        dep, base = mes
        exp = export_v4.ProcessExporterFortranMEGroup()
        with self.assertLogs('madgraph.export_v4', level='WARNING') as caught:
            cmap = exp._crossgroup_configmap(dep, base, 0)
        self.assertEqual(cmap, list(range(1, len(dep.get('diagrams')) + 1)),
                         'an unmatchable pair must fall back to the identity')
        said = '\n'.join(caught.output)
        for name in ('uux_ttxgg', 'uux_ttxuux'):
            self.assertIn(name, said,
                          'the fallback warning does not name %s:\n%s'
                          % (name, said))


class TestCrossingFlavorRepresentative(unittest.TestCase):
    """The PDG signature reported for a flavor index must be the signature of the
    flavor class that index actually selects.

    compute_crossing_pdg_entries reads the PDG table of _build_flav_pdg_tables,
    which has ONE ROW PER PHYSICAL FLAVOR COMBINATION, while its flavor index
    counts the coupling-equivalence classes of get_external_flavors_with_iden()
    -- the FLAVOR table the backends read is built from each class's
    representative flav[0]. Row f is the representative of class f only while the
    leading rows happen to BE the representatives. ``p p > j j`` with the
    crossings unfolded has ``Q Q~ > Q Q~``, whose three classes sit at rows 0, 1
    and 4: taking the ordinal names ``q q~ > q'' q~''`` (a member of class 1) for
    the class that is really ``q q~' > q q~'``. The routing decision, the
    recorded-crossing intersection behind crossed_flavors.dat and the C++
    demo_pdg table all match on exactly this signature.

    ``allowed_flavors_with_iden_pdgs`` is the independent oracle here: it carries
    the class representative's PDGs directly and shares no code with the table
    indexing under test."""

    def _mes(self, proc):
        import madgraph.iolibs.group_subprocs as group_subprocs
        import madgraph.iolibs.export_v4 as export_v4
        cmd = cmd_interface.MasterCmd()
        cmd.run_cmd('import model sm')
        # The default multi-flavor j is the point: with a single quark flavor
        # every class is a single row and the misalignment cannot appear.
        # Unfolded (MG_MERGE_CROSSING=off) so the crossed modules still exist,
        # exactly as TestCrossingPartition does.
        old = os.environ.get('MG_MERGE_CROSSING')
        os.environ['MG_MERGE_CROSSING'] = 'off'
        try:
            cmd.run_cmd('generate %s --use_crossing=True' % proc)
        finally:
            if old is None:
                os.environ.pop('MG_MERGE_CROSSING', None)
            else:
                os.environ['MG_MERGE_CROSSING'] = old
        groups = group_subprocs.SubProcessGroup.group_amplitudes(
            cmd._curr_amps, 'madevent')
        mes = []
        for g in groups:
            g.generate_matrix_elements()
            mes.extend(g.get('matrix_elements'))
        return mes, export_v4.ProcessExporterFortran()

    def test_identity_signature_is_the_class_representative(self):
        mes, exp = self._mes('p p > j j')
        self.assertTrue(mes)
        for me in mes:
            _classes, class_pdgs = \
                me.get_external_flavors_with_iden(return_pdgs=True)
            expected = [tuple(members[0]) for members in class_pdgs]
            got = [pdg for (_idx, cross, _flav, pdg) in
                   exp.compute_crossing_pdg_entries(me) if cross == 0]
            self.assertEqual(
                got, expected,
                'identity signatures of %s do not name its flavor classes'
                % me.get('processes')[0].shell_string())

    def test_fixture_exercises_a_misaligned_matrix_element(self):
        """Guard the test above from going toothless: if grouping ever stops
        producing a matrix element whose classes are NOT the leading rows, the
        assertion holds trivially and no longer covers the defect."""
        mes, exp = self._mes('p p > j j')
        misaligned = [me for me in mes
                      if exp._flavor_rep_rows(me)
                      != list(range(len(exp._flavor_rep_rows(me))))]
        self.assertTrue(
            misaligned,
            'no matrix element with a non-ordinal class representative; '
            'the representative test no longer covers the ordinal bug')


class TestCrossingReorderCandidates(unittest.TestCase):
    """find_reorder_candidates names the modules that keep their own matrix<i>.f
    only because one flavor class is listed with its final legs the other way
    round -- the modules a generation-time split could free.

    ``p p > j j`` unfolded has the canonical example: ``Q Q~ > Q Q~`` routes two
    of its three classes to ``Q Q > Q Q`` as generated, and is held back by the
    flavor-changing annihilation ``q q~ > q' q~'``, which the crossing delivers
    with the two light legs swapped. The module cannot relabel itself out of it
    (its leg pattern is shared by every row) and no single ordering suits all
    three classes, so the class has to be peeled into its own subprocess.

    This is analysis only: the second test pins that calling it does not move the
    routing, so it can be trusted not to change any output."""

    def _mes(self, proc):
        import madgraph.iolibs.group_subprocs as group_subprocs
        import madgraph.iolibs.export_v4 as export_v4
        cmd = cmd_interface.MasterCmd()
        cmd.run_cmd('import model sm')
        old = os.environ.get('MG_MERGE_CROSSING')
        os.environ['MG_MERGE_CROSSING'] = 'off'
        try:
            cmd.run_cmd('generate %s --use_crossing=True' % proc)
        finally:
            if old is None:
                os.environ.pop('MG_MERGE_CROSSING', None)
            else:
                os.environ['MG_MERGE_CROSSING'] = old
        groups = group_subprocs.SubProcessGroup.group_amplitudes(
            cmd._curr_amps, 'madevent')
        out = []
        for g in groups:
            g.generate_matrix_elements()
            mes = g.get('matrix_elements')
            if len(mes) > 1:
                out.append(mes)
        return out, export_v4.ProcessExporterFortran()

    def test_qqx_is_held_back_by_one_class(self):
        groups, exp = self._mes('p p > j j')
        found = []
        for mes in groups:
            names = [m.get('processes')[0].shell_string() for m in mes]
            bases, _routing = exp.partition_crossing_classes(mes)
            for i, peel in exp.find_reorder_candidates(mes).items():
                found.append((names[i], len(peel), peel))
                # a candidate must be a module that currently keeps its own ME
                self.assertIn(i, bases,
                              '%s is not a base; nothing to free' % names[i])
                nx, nini = mes[i].get_nexternal_ninitial()
                for _flav0, sigma, base_index, iflav in peel:
                    # sigma permutes FINAL legs only -- the beams are not
                    # interchangeable for the PDF
                    self.assertEqual(sorted(sigma), list(range(nx)))
                    self.assertEqual(list(sigma[:nini]), list(range(nini)))
                    self.assertNotEqual(tuple(sigma), tuple(range(nx)),
                                        'a candidate needs a real reorder')
                    self.assertIn(base_index, bases)
                    self.assertGreaterEqual(iflav, 1)
        self.assertTrue(found, 'no reorder candidate found in p p > j j; the '
                               'fixture no longer covers the split case')
        self.assertTrue(any(n.endswith('QQx_QQx') for n, _c, _p in found),
                        'expected Q Q~ > Q Q~ among the candidates: %s' % found)

    def test_detection_does_not_move_the_routing(self):
        """It is analysis: asking must not change what routing decides."""
        groups, exp = self._mes('p p > j j')
        for mes in groups:
            before = exp.partition_crossing_classes(mes)
            exp.find_reorder_candidates(mes)
            after = exp.partition_crossing_classes(mes)
            self.assertEqual(before, after)


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


class TestMadeventDecayChainCrossing(unittest.TestCase):
    """End-to-end: a decay-chain crossing routed through the base's SMATRIX in
    madevent gives the same cross section as an independent build.

    ``p p > w+ j, w+ > j j`` crosses the light partons of the production while
    the ``w+ > j j`` decay block rides along on the top-level W+; the crossed
    subprocesses (``g q~ > w+ q~``, ...) reuse the base matrix element through
    the crossing-aware SMATRIX (matrix2_router dispatches to SMATRIX1 with a
    crossed FLAV_IDX and rebuilds the crossed, resonance-level denominator). A
    ``--use_crossing=False`` build computes every subprocess independently
    instead. With the same seed the routed and the independent integration must
    agree -- a wrong crossed denominator, a split decay block, or a mis-routed
    flavor would move the cross section.

    Runs two full (small) madevent generations, so it is slow.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='cross_mev_dc_')

    def tearDown(self):
        if os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _xsec(self, options, name):
        from madgraph import MG5DIR
        outdir = pjoin(self.tmpdir, name)
        card = pjoin(self.tmpdir, 'cmd_%s.txt' % name)
        with open(card, 'w') as fsock:
            fsock.write('generate p p > w+ j, w+ > j j %s\n'
                        'output madevent %s -f\n'
                        'launch\n'
                        'set nevents 1000\n'
                        'set iseed 424242\n' % (options, outdir))
        subprocess.call([sys.executable, pjoin(MG5DIR, 'bin', 'mg5_aMC'), card])
        results = pjoin(outdir, 'SubProcesses', 'results.dat')
        self.assertTrue(os.path.isfile(results),
                        'madevent produced no results (%s)' % results)
        with open(results) as fsock:
            # results.dat: cross-section, abs error, ... (in pb).
            fields = fsock.readline().split()
        return float(fields[0]), float(fields[1])

    def test_decay_chain_crossing_xsec_matches(self):
        crossed, err_c = self._xsec('', 'on')
        independent, err_i = self._xsec('--use_crossing=False', 'off')
        self.assertGreater(independent, 0.0,
                           'independent build gives a null cross section')
        scale = max(abs(crossed), abs(independent), 1e-99)
        self.assertLessEqual(
            abs(crossed - independent) / scale, 1e-2,
            'p p > w+ j, w+ > j j crossing-routed xsec %r +- %r disagrees with '
            'the independent build %r +- %r'
            % (crossed, err_c, independent, err_i))


class TestMadeventInclusiveCrossingXsec(unittest.TestCase):
    """End-to-end: routing crossed subprocesses through a shared base matrix
    element must not move the INCLUSIVE cross section.

    The plain (no decay chain) counterpart of TestMadeventDecayChainCrossing,
    and the configuration where the crossing router has the most to get wrong.
    With flavor grouping ``p p > t t~ j j`` collapses to five subprocess groups,
    and two of them -- gq_ttxgq and qq_ttxqq -- are served by a cross-GROUP
    router: they carry a ``matrix<i>_router.f`` (plus ``crossgroup_helunion.dat``
    and ``crossgroup.mk``) instead of their own matrix element, i.e. their
    flavors are evaluated by ANOTHER group's matrix element under a crossing,
    over the helicity union of the two groups. Nothing else in the suite
    integrates that path -- Track B is exercised at the matrix-element level
    only.

    The summed cross section is what catches it. A wrong crossed averaging
    denominator, multi-channel row or good-helicity union leaves the per-flavor
    matrix elements agreeing (those are compared in
    TestStandaloneMadeventMatrixElementConsistency) while moving the integral,
    which is exactly how the routed groups lost ~29% before the helicity union
    was fed to the Track-A routers.

    Runs two full madevent integrations, but the flavor grouping keeps them
    small: ~40s each. Reference numbers at the time of writing --
    416.6 +- 2.4 pb routed vs 413.7 +- 2.6 pb independent, i.e. 0.8 sigma apart.
    """

    PROCESS = 'p p > t t~ j j'
    NEVENTS = 1000
    SEED = 191919

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='cross_mev_ttjj_')

    def tearDown(self):
        if os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _generate_and_integrate(self, options, name):
        """Generate + integrate the process; return (outdir, xsec, error) in pb."""
        from madgraph import MG5DIR
        outdir = pjoin(self.tmpdir, name)
        card = pjoin(self.tmpdir, 'cmd_%s.txt' % name)
        with open(card, 'w') as fsock:
            fsock.write('generate %s %s\n'
                        'output madevent %s -f\n'
                        'launch\n'
                        'set nevents %d\n'
                        'set iseed %d\n'
                        % (self.PROCESS, options, outdir,
                           self.NEVENTS, self.SEED))
        subprocess.call([sys.executable, pjoin(MG5DIR, 'bin', 'mg5_aMC'), card])
        results = pjoin(outdir, 'SubProcesses', 'results.dat')
        self.assertTrue(
            os.path.isfile(results),
            'madevent produced no results for %s (%s)'
            % (options or 'the default (crossing on) build', results))
        with open(results) as fsock:
            # results.dat: cross-section, abs error, ... (in pb).
            fields = fsock.readline().split()
        return outdir, float(fields[0]), float(fields[1])

    @staticmethod
    def _routed_groups(outdir):
        """The subprocess groups served by a cross-group crossing router."""
        subproc = pjoin(outdir, 'SubProcesses')
        routed = []
        for name in sorted(os.listdir(subproc)):
            pdir = pjoin(subproc, name)
            if not name.startswith('P') or not os.path.isdir(pdir):
                continue
            if any(re.match(r'matrix\d+_router\.f$', entry)
                   for entry in os.listdir(pdir)):
                routed.append(name)
        return routed

    def test_inclusive_crossing_xsec_matches(self):
        crossed_dir, crossed, err_c = self._generate_and_integrate('', 'on')
        independent_dir, independent, err_i = self._generate_and_integrate(
            '--use_crossing=False', 'off')

        # Guard the premise: the default build must really evaluate some group
        # through another group's matrix element, and the reference build must
        # not -- otherwise this compares two identical builds and can never fail.
        routed = self._routed_groups(crossed_dir)
        self.assertTrue(
            routed, 'no subprocess group is served by a crossing router, so the '
            'comparison would be between two identical builds')
        self.assertEqual(
            self._routed_groups(independent_dir), [],
            '--use_crossing=False still emitted a crossing router')

        self.assertGreater(independent, 0.0,
                           'the independent build gives a null cross section')
        # Same seed and the same channels, so the two runs must agree well
        # inside their combined statistical error; the 1% floor absorbs the grid
        # noise the different routing can introduce.
        tolerance = max(1e-2 * independent, 3.0 * math.hypot(err_c, err_i))
        self.assertLessEqual(
            abs(crossed - independent), tolerance,
            '%s crossing-routed xsec %r +- %r disagrees with the independent '
            'build %r +- %r (groups routed through a crossing: %s)'
            % (self.PROCESS, crossed, err_c, independent, err_i,
               ', '.join(routed)))


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

    def test_color_flow_code_round_trip(self):
        """decode(code(flow)) reproduces the flow's canonical connectivity, and
        the slot structure is FLOW-INDEPENDENT (it is process data, fixed by the
        colour representations). Together these are what allow the colour tags
        to be rebuilt from the code alone instead of read out of the generated
        ICOLUP table -- the step this encoding is aiming at.
        """
        import madgraph.core.helas_objects as helas_objects
        import madgraph.iolibs.export_v4 as export_v4
        exp = export_v4.ProcessExporterFortranMEGroup.__new__(
            export_v4.ProcessExporterFortranMEGroup)
        checked = 0
        for proc, _nflow in self.PROCS:
            cmd = cmd_interface.MasterCmd()
            cmd.exec_cmd('generate %s' % proc, printcmd=False)
            me = helas_objects.HelasMultiProcess(cmd._curr_amps)
            for m in me.get('matrix_elements'):
                if not m.get('color_basis'):
                    continue
                states = [l.get('state') for l in
                          m.get('processes')[0].get_legs_with_decays()]
                slots = None
                for flow in exp._module_color_flows(m):
                    conns = exp._color_flow_canon(flow, states)
                    this = exp._color_flow_slots(conns)
                    if slots is None:
                        slots = this
                    # the slot structure must not depend on the flow
                    self.assertEqual(this, slots,
                                     '%s: slot structure varies between flows '
                                     '(%s vs %s)' % (proc, this, slots))
                    code = exp._color_flow_code(conns)
                    self.assertIsNotNone(code, '%s: flow did not encode' % proc)
                    back = exp._color_flow_decode(code, slots[0], slots[1])
                    self.assertEqual(back, conns,
                                     '%s: code %d does not round-trip\n  got %s'
                                     '\n  want %s'
                                     % (proc, code, sorted(back), sorted(conns)))
                    checked += 1
        self.assertTrue(checked, 'no colour flow was round-tripped')


class TestMadeventColorFlowRatio(unittest.TestCase):
    """End-to-end guard on the COLOUR written to the LHE, for u u~ > u u~.

    Every event's colour tags must form a clean colour<->anticolour bijection
    once the initial-state legs swap roles (the canonical form the colour-flow
    code is built on): each colour label is matched by exactly one anticolour
    label. That is the colour analogue of "the helicity is one of the physical
    states", and it is what breaks first if the colour flow written to the event
    is ever rebuilt wrongly -- e.g. when the tags start being decoded from the
    canonical colour-flow code instead of read from the ICOLUP table.

    u u~ > u u~ is chosen deliberately: its two colour flows are STRONGLY
    asymmetric (~98/2), so the test can pin down WHICH flow is which. A process
    whose flows are related by a symmetry -- g g > t t~ splits 50/50 -- would
    pass just as happily with the two flow labels SWAPPED, which is exactly the
    bug this is meant to catch. Here a swap inverts 98/2 into 2/98.

    The dominant flow is identified topologically (do the colour connections
    stay inside the initial/final groups, or cross between them?) rather than by
    raw leg indices, so the check does not depend on leg ordering. Runs a small
    madevent generation, so it is a slow test.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='cross_col_ratio_')

    def tearDown(self):
        if os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    @staticmethod
    def _canon(parts):
        """{colour label: [legs]}, {anticolour label: [legs]} with initial-state
        legs swapping the two roles."""
        col, anti = {}, {}
        for i, p in enumerate(parts):
            c, a = int(p.color1), int(p.color2)
            if p.status == -1:
                c, a = a, c
            if c:
                col.setdefault(c, []).append(i)
            if a:
                anti.setdefault(a, []).append(i)
        return col, anti

    def test_color_flow_ratio_uux_uux(self):
        from madgraph import MG5DIR
        from madgraph.various import lhe_parser
        outdir = pjoin(self.tmpdir, 'uux')
        card = pjoin(self.tmpdir, 'cmd.txt')
        with open(card, 'w') as f:
            f.write('generate u u~ > u u~\n'
                    'output madevent %s -f\n'
                    'launch\n'
                    'set nevents 2000\n'
                    'set iseed 909\n' % outdir)
        subprocess.call([sys.executable, pjoin(MG5DIR, 'bin', 'mg5_aMC'), card])

        lhe = pjoin(outdir, 'Events', 'run_01', 'unweighted_events.lhe.gz')
        self.assertTrue(os.path.isfile(lhe),
                        'madevent produced no LHE file (%s)' % lhe)

        nevt = 0
        sigs = {}
        for event in lhe_parser.EventFile(lhe):
            parts = [p for p in event]
            nevt += 1
            col, anti = self._canon(parts)
            # (1) structure: a perfect colour <-> anticolour matching
            self.assertEqual(set(col), set(anti),
                             'colour labels do not pair with anticolour labels '
                             '(event %d): %s vs %s' % (nevt, sorted(col),
                                                       sorted(anti)))
            self.assertTrue(col, 'event %d carries no colour at all' % nevt)
            for lbl, legs in col.items():
                self.assertEqual(len(legs), 1,
                                 'colour label %s appears on %d legs (event %d)'
                                 % (lbl, len(legs), nevt))
                self.assertEqual(len(anti[lbl]), 1,
                                 'anticolour label %s appears on %d legs '
                                 '(event %d)' % (lbl, len(anti[lbl]), nevt))
            # topological signature of the flow: does each colour connection
            # stay inside the initial / final group, or cross between them?
            ini = set(i for i, p in enumerate(parts) if p.status == -1)
            sig = tuple(sorted(('I' if c in ini else 'F')
                               + ('I' if a in ini else 'F')
                               for c, a in ((col[l][0], anti[l][0])
                                            for l in col)))
            sigs[sig] = sigs.get(sig, 0) + 1

        self.assertGreater(nevt, 100, 'too few events generated (%d)' % nevt)
        # (2) exactly the two expected colour topologies
        self.assertEqual(set(sigs), {('FF', 'II'), ('FI', 'IF')},
                         'unexpected colour-flow topologies: %s' % sigs)
        same = sigs[('FF', 'II')] / nevt     # connections inside each group
        cross = sigs[('FI', 'IF')] / nevt    # connections crossing the groups
        # (3) the asymmetry, and crucially WHICH topology dominates: swapping
        # the two flow labels would invert this and fail here.
        self.assertGreater(same, 0.9,
                           'the initial-initial / final-final colour topology '
                           'should dominate u u~ > u u~ (measured ~0.98), got '
                           '%.3f (cross=%.3f)' % (same, cross))
        self.assertTrue(0.002 < cross < 0.1,
                        'the crossing colour topology should be present but '
                        'strongly suppressed (measured ~0.02), got %.4f'
                        % cross)


class TestMadeventRouterColorSelection(unittest.TestCase):
    """A within-group (Track A) router must RESELECT the colour flow, with its
    OWN colour-config mask -- not relabel the flow its base picked.

    A router has no matrix element of its own: it calls the base SMATRIX with a
    crossed FLAV_IDX. That base runs SELECT_COLOR before it returns, masking its
    JAMP2 with the BASE's ICOLAMP row. ICOLAMP is indexed by (flow, config,
    SUBPROCESS), and two subprocesses of one group do not have the same row: in
    ``g u > g u`` / ``g u~ > g u~`` the rows for configs 2 and 3 are swapped, so
    at the same live ICONFIG the base allows exactly the flow the router's own
    SELECT_COLOR forbids. Whatever the router then does with that index -- even
    the identity, which is what a crossing-covariant flow ORDER gives -- the
    event carries a colour topology the module would never have chosen.

    The fix is to discard the base's choice and reselect: permute the base's
    published per-flow JAMP2 (COMMON/TO_XG_JAMP2) into this subprocess's flow
    order and call SELECT_COLOR with the ROUTER's proc_id (XG_SELCOL<i>). This
    is the same thing the cross-group path (Track B) already does.

    Checked twice over.  test_router_reselects_colour_with_its_own_mask is the
    structural half: it reads the generated fortran, and -- crucially -- asserts
    that some router really does have a different ICOLAMP row from its base, so
    the guard cannot go vacuous if the diagram numbering ever becomes
    crossing-covariant.  test_router_colour_topology_matches_no_crossing is the
    behavioural half, and the only kind of check that catches this class of bug:
    the cross section agreed to 0.02% while ~10% of the affected class carried
    the wrong flow, and per-point SMATRIX probes run before the good-helicity
    state warms up, a regime production never reaches.  So it compares the
    COLOUR TOPOLOGY DISTRIBUTION of two full event samples, one routed and one
    built with --use_crossing=False.

    That comparison is run over two canonical forms, because the colour-only one
    has a structural blind spot.  Canonicalising a topology means minimising it
    over every relabelling of the legs, and legs may only be exchanged when they
    have the same TYPE.  With the type (status, pid) the two gluons of
    g g > q q~ are interchangeable, so the minimisation swaps them freely and
    maps that class's two colour flows onto each other: both collapse into ONE
    category, and NO redistribution between them can ever be detected.  Adding
    the helicity to the type -- (status, pid, helicity) -- pins the permutation
    whenever the gluons differ in helicity and separates the flows again.  That
    refinement is what exposed a crossing build assigning ~10% of g g > q q~ a
    colour flow drawn ~50/50 instead of from JAMP2: the recycled optim of a
    crossing BASE kept every helicity config instead of the good-hel union, and
    the configs with |M|^2 == 0 still carry non-zero individual diagrams and
    JAMPs, which silently reweighted the AMP2 channel weights and the JAMP2
    colour weights.  Marginal helicity, marginal colour and the cross section
    were all correct while that was happening; only the correlation moved.

    ``g u u~`` dijets rather than ``p p > j j``: same subprocess groups, same
    routers, one quark flavour instead of four, so a generation takes seconds.
    """

    DEFINE = 'define q1 = g u u~'
    PROCESS = 'q1 q1 > q1 q1'
    NEVENTS = 400000
    SEED = 777
    # A flavour class needs this many reference events before its topology
    # fractions are compared. At 5000 the statistical error on a fraction is
    # 0.7%, an order of magnitude below the shift being looked for.
    MIN_CLASS = 5000
    # The class the within-group router serves here: u~ g > u~ g, evaluated by
    # the u g > u g matrix element under a crossing. Named explicitly because it
    # is the only class in this process whose colour selection the router
    # decides, and g g > g g outnumbers it many times over -- a comparison that
    # quietly stopped reaching it would pass no matter what the router did.
    ROUTED_CLASS = ((-2, 21), (-2, 21))
    # Tolerated shift of a topology fraction, on top of a 4 sigma statistical
    # allowance. The defect this guards moves it by ~3 points (0.403 -> 0.435 on
    # u~ g > u~ g at these beams, 8 sigma); the fix leaves it inside 1 sigma.
    MAX_SHIFT = 0.015
    # Significance of the homogeneity chi-square (see _homogeneity), used by
    # TestMadeventCrossingBaseColorFlow rather than by this class. 4 sigma
    # (p ~ 3e-5) keeps a spurious failure rare while leaving a wide margin on
    # the defect it guards: measured 0.4 on 3 dof fixed, 24.5 critical.
    CHI2_Z = 4.0

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='cross_router_col_')

    def tearDown(self):
        if os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _generate(self, options, name, launch=False):
        """Generate (and optionally integrate) the process; return its outdir."""
        from madgraph import MG5DIR
        outdir = pjoin(self.tmpdir, name)
        card = pjoin(self.tmpdir, 'cmd_%s.txt' % name)
        lines = ['%s\n' % self.DEFINE,
                 'generate %s %s\n' % (self.PROCESS, options),
                 'output madevent %s -f -nojpeg\n' % outdir]
        if launch:
            lines += ['launch\n',
                      'set nevents %d\n' % self.NEVENTS,
                      'set iseed %d\n' % self.SEED,
                      # a broken local lhapdf kills the systematics step, and
                      # this test has no use for the reweighting anyway
                      'set use_syst False\n',
                      # Beam 2 an ANTIproton: the routed subprocess is
                      # g u~ > g u~, so on p p it is a sea channel and gets ~4%
                      # of the events. Against an antiproton the u~ is valence
                      # and the class doubles, which is what buys the routed
                      # class the statistics to resolve the shift without
                      # doubling the runtime. Nothing else about the test
                      # depends on the beams.
                      'set lpp2 -1\n']
        with open(card, 'w') as fsock:
            fsock.writelines(lines)
        subprocess.call([sys.executable, pjoin(MG5DIR, 'bin', 'mg5_aMC'), card])
        self.assertTrue(os.path.isdir(pjoin(outdir, 'SubProcesses')),
                        'madevent produced no output for %r' % (options or
                                                                'the default'))
        return outdir

    @staticmethod
    def _flat(path):
        """The file's code with comments, continuations and all whitespace gone,
        so a pattern can be matched without caring how the writer wrapped it."""
        out = []
        with open(path) as fsock:
            for line in fsock:
                if not line.strip() or line[0] in 'Cc*!':
                    continue
                body = line[6:] if len(line) > 6 else ''
                if len(line) > 5 and line[5] not in ' \t':
                    out.append(body)        # continuation of the previous line
                else:
                    out.append('\n' + body)
        return re.sub(r'[ \t]', '', ''.join(out))

    @classmethod
    def _routers(cls, outdir):
        """{P directory: {router proc_id: base proc_id}} for every Track A router."""
        subproc = pjoin(outdir, 'SubProcesses')
        found = {}
        for name in sorted(os.listdir(subproc)):
            pdir = pjoin(subproc, name)
            if not name.startswith('P') or not os.path.isdir(pdir):
                continue
            for entry in sorted(os.listdir(pdir)):
                match = re.match(r'matrix(\d+)_router\.f$', entry)
                if not match:
                    continue
                bases = set(re.findall(r'CALLSMATRIX(\d+)\(',
                                       cls._flat(pjoin(pdir, entry))))
                # partition_crossing_classes only ever routes a module to a
                # single base, so this is one entry per router.
                found.setdefault(name, {})[int(match.group(1))] = \
                    int(bases.pop()) if len(bases) == 1 else None
        return found

    @staticmethod
    def _icolamp(pdir):
        """{proc_id: {config: (flow allowed, ...)}} out of coloramps.inc.

        Configs the file does not list are forbidden for every flow, which is
        exactly how the fortran DATA leaves them.
        """
        rows = {}
        text = ''
        with open(pjoin(pdir, 'coloramps.inc')) as fsock:
            for line in fsock:
                if len(line) > 5 and line[5] not in ' \t':
                    text += line[6:]
                else:
                    text += '\n' + line[6:] if len(line) > 6 else '\n'
        for stmt in text.split('\n'):
            match = re.match(r'\s*DATA\s*\(\s*ICOLAMP\(I,(\d+),(\d+)\)\s*,'
                             r'\s*I\s*=\s*1\s*,\s*(\d+)\s*\)\s*/(.*)/\s*$',
                             stmt.replace(' ', ''))
            if not match:
                continue
            iconfig, iproc = int(match.group(1)), int(match.group(2))
            vals = tuple(v.strip().upper().startswith('.T')
                         for v in match.group(4).split(','))
            rows.setdefault(iproc, {})[iconfig] = vals
        return rows

    @classmethod
    def _mismatched_masks(cls, outdir):
        """(P dir, router, base) for every router whose ICOLAMP row differs from
        its base's -- i.e. every router the base's colour choice would mislead."""
        out = []
        for pname, pairs in cls._routers(outdir).items():
            rows = cls._icolamp(pjoin(outdir, 'SubProcesses', pname))
            for router, base in sorted(pairs.items()):
                if base is None:
                    continue
                if rows.get(router, {}) != rows.get(base, {}):
                    out.append((pname, router, base))
        return out

    def test_router_reselects_colour_with_its_own_mask(self):
        outdir = self._generate('', 'struct')
        routers = self._routers(outdir)
        self.assertTrue(routers,
                        '%s produced no within-group crossing router, so this '
                        'test would check nothing' % self.PROCESS)

        # The premise: at least one router really is masked differently from its
        # base. Without this the whole comparison is between two ways of writing
        # the same answer and could never fail.
        mismatched = self._mismatched_masks(outdir)
        self.assertTrue(
            mismatched,
            'no router has an ICOLAMP row different from its base\'s, so '
            'reselecting colour could not change any event -- the guard below '
            'has become vacuous and needs a process where it bites (routers '
            'found: %s)' % routers)

        for pname, pairs in sorted(routers.items()):
            pdir = pjoin(outdir, 'SubProcesses', pname)
            for router, base in sorted(pairs.items()):
                self.assertIsNotNone(
                    base, 'matrix%d_router.f in %s dispatches to more than one '
                    'base SMATRIX' % (router, pname))
                code = self._flat(pjoin(pdir, 'matrix%d_router.f' % router))
                # (1) the helper exists and masks with the ROUTER's own proc_id
                self.assertIn('SUBROUTINEXG_SELCOL%d(RCOL,IFLAV,IVEC,ICOL)'
                              % router, code,
                              'matrix%d_router.f (%s) has no colour-reselection '
                              'helper' % (router, pname))
                self.assertIn('CALLSELECT_COLOR(RCOL,JD,ICONFIG,%d,ICOL,IVEC)'
                              % router, code,
                              'XG_SELCOL%d (%s) does not run SELECT_COLOR with '
                              'its own subprocess index as IPROC, so it masks '
                              'the flows with another subprocess\'s ICOLAMP row'
                              % (router, pname))
                # (2) every dispatched flavour goes through it -- an identity
                # flow order is NOT a reason to keep the base's pick
                ncall = len(re.findall(r'CALLSMATRIX%d\(' % base, code))
                nsel = len(re.findall(r'CALLXG_SELCOL%d\(' % router, code))
                self.assertEqual(
                    nsel, ncall,
                    'matrix%d_router.f (%s) reselects colour for %d of its %d '
                    'routed flavours' % (router, pname, nsel, ncall))
                # (3) nothing relabels the base's own selection any more
                self.assertNotIn('ICOL=COLMAP_', code,
                                 'matrix%d_router.f (%s) still relabels the '
                                 'base\'s colour index' % (router, pname))
                self.assertNotIn('IF(XDCD(XCK).EQ.XCNEW)ICOL=XCK', code,
                                 'matrix%d_router.f (%s) still translates the '
                                 'base\'s colour index through the flow code'
                                 % (router, pname))
                # (4) the base has to publish the per-flow JAMP2 the helper reads
                candidates = [pjoin(pdir, 'matrix%d_orig.f' % base),
                              pjoin(pdir, 'matrix%d.f' % base)]
                bfile = [c for c in candidates if os.path.isfile(c)]
                self.assertTrue(bfile, 'no source for base SMATRIX%d in %s'
                                % (base, pname))
                bcode = self._flat(bfile[0])
                self.assertIn('COMMON/TO_XG_JAMP2/XG_JAMP2', bcode,
                              '%s does not publish its per-flow JAMP2, so '
                              'XG_SELCOL%d has nothing to reselect from'
                              % (os.path.basename(bfile[0]), router))
                self.assertIn('XG_JAMP2(I,IVEC)=JAMP2(I)', bcode,
                              '%s declares TO_XG_JAMP2 but never fills it'
                              % os.path.basename(bfile[0]))

    def test_router_colour_topology_matches_no_crossing(self):
        from madgraph.various import lhe_parser

        routed = self._generate('', 'on', launch=True)
        plain = self._generate('--use_crossing=False', 'off', launch=True)

        self.assertTrue(
            self._mismatched_masks(routed),
            'the routed build has no router masked differently from its base, '
            'so this comparison cannot fail')
        self.assertEqual(self._routers(plain), {},
                         '--use_crossing=False still emitted a crossing router')

        ref = self._topologies(plain, lhe_parser)
        got = self._topologies(routed, lhe_parser)
        nall = sum(sum(c['colour'].values()) for c in ref.values())
        self.assertGreater(nall, 0,
                           'the --use_crossing=False build produced no events')
        # The launch has to have honoured `set nevents`: at the run_card default
        # the routed class falls below MIN_CLASS, every class but g g > g g is
        # skipped and the comparison silently checks nothing.
        self.assertGreaterEqual(
            nall, 0.9 * self.NEVENTS,
            'the --use_crossing=False build wrote %d events, not the %d asked '
            'for -- the per-class statistics this test needs are not there'
            % (nall, self.NEVENTS))

        compared = []
        for flav in sorted(ref):
            nref = sum(ref[flav]['colour'].values())
            ngot = sum(got.get(flav, {}).get('colour', {}).values())
            logger.info('  %-18s %7d ref %7d routed   %s', self._fmt(flav),
                        nref, ngot,
                        ' '.join('%.4f/%.4f' % (
                            got.get(flav, {}).get('colour', {}).get(t, 0)
                            / float(ngot or 1),
                            ref[flav]['colour'][t] / float(nref))
                            for t in sorted(ref[flav]['colour'])))
            if nref < self.MIN_CLASS or not ngot:
                continue
            compared.append(flav)
            # Both observables, weakest first. 'colour' is what a wrong ICOLAMP
            # row moves; 'joint' additionally catches anything that moves the
            # flow WITHIN a helicity configuration, which for a class with two
            # identical gluons is the only thing there is to see.
            for obs in ('colour', 'joint'):
                rbin, gbin = ref[flav][obs], got[flav][obs]
                # (a) as specified: no category the reference never produces
                extra = [t for t in gbin
                         if t not in rbin
                         and nref * gbin[t] / float(ngot) >= 5.0]
                self.assertFalse(
                    extra,
                    '%s: the routed build writes %d %s category(ies) the '
                    '--use_crossing=False build never produces (%s)'
                    % (self._fmt(flav), len(extra), obs,
                       ', '.join('%d events' % gbin[t] for t in extra)))
                # (b) and, strictly stronger, the same MIX of them: a wrong
                # ICOLAMP row moves weight between categories both builds can
                # produce, so (a) alone does not see it.
                for topo in set(list(rbin) + list(gbin)):
                    pref = rbin.get(topo, 0) / float(nref)
                    pgot = gbin.get(topo, 0) / float(ngot)
                    sigma = math.sqrt(pref * (1 - pref) / nref
                                      + pgot * (1 - pgot) / ngot)
                    self.assertLessEqual(
                        abs(pgot - pref), max(self.MAX_SHIFT, 4.0 * sigma),
                        '%s: %s category %s carries %.4f of the class in the '
                        'routed build but %.4f in the --use_crossing=False '
                        'build (%d vs %d events, %.1f sigma) -- the crossing '
                        'build is not choosing the flow the module itself would'
                        % (self._fmt(flav), obs, topo, pgot, pref,
                           gbin.get(topo, 0), rbin.get(topo, 0),
                           abs(pgot - pref) / sigma if sigma else 0.0))
                # Deliberately NOT the homogeneity chi-square here, though
                # _homogeneity is what TestMadeventCrossingBaseColorFlow uses.
                # g g > g g carries 325k of the 400k events in this process, and
                # at that size a chi-square resolves differences far below the
                # MAX_SHIFT floor this test was calibrated around -- it would be
                # a much tighter bar than intended on the classes it was never
                # meant to police. The sharp statistic belongs on the class it
                # was measured on.
        # The comparison is only worth anything if it reached the class the
        # router actually serves; without this it degrades to g g > g g, which
        # no router touches, and passes whatever the routers do.
        self.assertIn(
            self.ROUTED_CLASS, compared,
            '%s -- the class the within-group router serves -- was not among '
            'the %d compared (%s), so this test checked nothing about the '
            'router' % (self._fmt(self.ROUTED_CLASS), len(compared),
                        ', '.join(self._fmt(f) for f in compared)))
        # The identical-gluon class g g > u u~ is deliberately NOT required
        # here: g g > g g takes 81% of this process and starves it to 0.5%
        # (2139 events in 400k), which is an order of magnitude short of what
        # it takes to resolve a flow shift inside it. TestMadeventCrossingBase-
        # ColorFlow covers that class on a process where it is not starved.

    @staticmethod
    def _fmt(flav):
        return '%s > %s' % (' '.join(str(p) for p in flav[0]),
                            ' '.join(str(p) for p in flav[1]))

    @staticmethod
    def _homogeneity(ref, got):
        """(chi2, dof, critical value) for 'both samples share one category mix'.

        The per-category threshold above asks each category on its own to move by
        more than max(MAX_SHIFT, 4 sigma). That is the right shape for a flow
        that lands in the wrong bucket outright, but it has little power against
        a COHERENT redistribution: the shift is divided among the categories and
        each piece stays under the bar while the pattern as a whole is far from
        chance. This is the standard 2 x K homogeneity chi-square on the raw
        counts, which aggregates exactly that pattern.

        Critical value is the Wilson-Hilferty quantile at CHI2_Z, so no scipy.
        """
        cats = set(list(ref) + list(got))
        nref, ngot = sum(ref.values()), sum(got.values())
        tot = float(nref + ngot)
        chi2, nbin = 0.0, 0
        for cat in cats:
            oref, ogot = ref.get(cat, 0), got.get(cat, 0)
            row = oref + ogot
            if not row:
                continue
            nbin += 1
            eref, egot = row * nref / tot, row * ngot / tot
            chi2 += (oref - eref) ** 2 / eref + (ogot - egot) ** 2 / egot
        dof = max(nbin - 1, 1)
        crit = dof * (1 - 2.0 / (9 * dof)
                      + TestMadeventRouterColorSelection.CHI2_Z
                      * math.sqrt(2.0 / (9 * dof))) ** 3
        return chi2, dof, crit

    @classmethod
    def _topologies(cls, outdir, lhe_parser):
        """{flavour class: {observable: {canonical category: events}}}.

        Two observables per event, both canonicalised the same way (see
        _canon_topology): 'colour' is the colour topology alone, 'joint' is the
        colour topology with each leg additionally typed by its HELICITY.
        'joint' is strictly finer, and for a class with two identical gluons it
        is the only one that separates the flows at all -- see the class
        docstring.
        """
        lhe = pjoin(outdir, 'Events', 'run_01', 'unweighted_events.lhe.gz')
        out = {}
        cache = {}
        for event in lhe_parser.EventFile(lhe):
            parts = [(int(p.status), int(p.pid), int(p.color1), int(p.color2),
                      int(p.helicity)) for p in event]
            key = tuple(parts)
            if key not in cache:
                flav = (tuple(sorted(p[1] for p in parts if p[0] == -1)),
                        tuple(sorted(p[1] for p in parts if p[0] == 1)))
                cache[key] = (flav, cls._canon_topology(parts),
                              cls._canon_topology(parts, helicity=True))
            flav, topo, joint = cache[key]
            bucket = out.setdefault(flav, {'colour': {}, 'joint': {}})
            bucket['colour'][topo] = bucket['colour'].get(topo, 0) + 1
            bucket['joint'][joint] = bucket['joint'].get(joint, 0) + 1
        return out

    @staticmethod
    def _canon_topology(parts, helicity=False):
        """Colour topology of one event, free of the leg-ordering convention.

        The connections are (leg holding a colour, leg holding the matching
        anticolour) with initial-state legs swapping the two roles -- the LHE
        runs an initial colour line 'through' the event, so without the swap a
        label sits in the same slot on two legs and the flow is not a bijection
        (the same canonical form _color_flow_canon uses in the exporter). The
        result is then minimised over every relabelling of the legs, so two
        modules that write the same physical flow in a different leg order give
        the same answer.

        The minimisation is only allowed to move legs of the same TYPE, and the
        type is what decides how much the canonical form can still see. With
        helicity=False the type is (status, pid), so two identical gluons are
        interchangeable and the minimisation is free to swap them -- which maps
        the two colour flows of g g > q q~ onto each other and collapses them
        into a single category, making any redistribution between them
        invisible. With helicity=True the type is (status, pid, helicity),
        which pins the permutation whenever the two gluons differ in helicity
        and keeps the flows apart.
        """
        col, anti = {}, {}
        for i, (status, _pid, c, a, _h) in enumerate(parts):
            if status == -1:
                c, a = a, c
            if c:
                col.setdefault(c, []).append(i)
            if a:
                anti.setdefault(a, []).append(i)
        conns = set()
        for label in set(list(col) + list(anti)):
            for cc, aa in zip(sorted(col.get(label, [])),
                              sorted(anti.get(label, []))):
                conns.add((cc, aa))
        if helicity:
            types = [(p[0], p[1], p[4]) for p in parts]
        else:
            types = [(p[0], p[1]) for p in parts]
        nleg = len(parts)
        best = None
        for perm in itertools.permutations(range(nleg)):
            inv = [0] * nleg
            for new, old in enumerate(perm):
                inv[old] = new
            cand = (tuple(types[old] for old in perm),
                    tuple(sorted((inv[i], inv[j]) for (i, j) in conns)))
            if best is None or cand < best:
                best = cand
        return best


class TestMadeventCrossingFinalLegSplit(unittest.TestCase):
    """MG_SPLIT_CROSSING peels the one flavor class that keeps a merged module
    compiled, into a sibling GENERATED with its final legs the other way round.

    ``Q Q~ > Q Q~`` bundles three coupling classes and drops its own matrix
    element only if EVERY one of them routes. Two do; the flavour-changing
    annihilation ``q q~ > q' q~'`` does not, because the crossing that reaches
    it off ``Q Q > Q Q`` (I=0/J=5) delivers the two light legs as ``(q~', q')``
    while the module lists ``(q', q~')``. A module cannot list one class
    differently -- its leg pattern is shared by every row -- so the class is
    peeled into a sibling with the swapped pattern and the two modules are given
    COMPLEMENTARY halves of the flavors.

    ``q q > q q`` with ``q = u d u~ d~`` rather than ``p p > j j``: same group,
    same peel, no gluon subprocesses, so a generation takes seconds.

    What is pinned here is what fails SILENTLY:

    * the halves must partition the flavors -- no combination covered twice (a
      double count, wrong by a factor 2) and none dropped. This is the assertion
      that catches IdentifyMETag re-merging the two modules: that tag identifies
      processes agreeing up to a LEG PERMUTATION, which is exactly what the two
      halves are, and merging them relabels one into the other's leg order and
      undoes the split with nothing to show for it.
    * the peel must actually eliminate a compiled matrix element, or the whole
      feature is cost without benefit.
    * it must not fire for an exporter that cannot consume a split pattern; mg7
      builds one module per leg pattern and dies with "no valid flavor
      configurations found for diagram 2" on the half that no longer has them.

    The colour/helicity correctness of the routed events is NOT checked here --
    that needs event samples, and TestMadeventRouterColorSelection is where that
    kind of comparison lives.
    """

    DEFINE = 'define q = u d u~ d~'
    PROCESS = 'q q > q q'

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='cross_split_')

    def tearDown(self):
        if os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _generate(self, name, split, fmt='madevent', options=''):
        from madgraph import MG5DIR
        outdir = pjoin(self.tmpdir, name)
        card = pjoin(self.tmpdir, 'cmd_%s.txt' % name)
        with open(card, 'w') as fsock:
            fsock.writelines(['%s\n' % self.DEFINE,
                              'generate %s %s\n' % (self.PROCESS, options),
                              'output %s %s -f -nojpeg\n' % (fmt, outdir)])
        env = dict(os.environ)
        env['MG_SPLIT_CROSSING'] = 'on' if split else ''
        subprocess.call([sys.executable, pjoin(MG5DIR, 'bin', 'mg5_aMC'), card],
                        env=env)
        return outdir

    @staticmethod
    def _counts(pdir):
        """(compiled matrix elements, crossing routers) in a P directory."""
        entries = os.listdir(pdir)
        return (len([e for e in entries
                     if re.match(r'matrix\d+(_orig)?\.f$', e)]),
                len([e for e in entries if re.match(r'matrix\d+_router\.f$', e)]))

    @staticmethod
    def _leshouche(pdir):
        """{subprocess: [IDUP row, ...]} out of leshouche.inc."""
        rows = {}
        with open(pjoin(pdir, 'leshouche.inc')) as fsock:
            for line in fsock:
                match = re.match(r'\s*DATA\s*\(IDUP\(I,(\d+),(\d+)\)\s*,'
                                 r'\s*I\s*=\s*1\s*,\s*(\d+)\s*\)\s*/([^/]*)/',
                                 line.replace(' ', ''))
                if match:
                    rows.setdefault(int(match.group(2)), []).append(
                        tuple(int(v) for v in match.group(4).split(',')))
        return rows

    @classmethod
    def _physical(cls, pdir, nini=2):
        """Counter of the PHYSICAL (initial, final) flavor combinations the
        directory covers, blind to the order the legs are listed in -- which is
        precisely what the two halves disagree about on purpose."""
        seen = {}
        for rows in cls._leshouche(pdir).values():
            for row in rows:
                key = (tuple(sorted(row[:nini])), tuple(sorted(row[nini:])))
                seen[key] = seen.get(key, 0) + 1
        return seen

    def test_split_partitions_the_flavors_and_frees_a_matrix_element(self):
        plain = self._generate('plain', split=False,
                               options='--use_crossing=False')
        split = self._generate('split', split=True)

        pdir_plain = pjoin(plain, 'SubProcesses', 'P1_qq_qq')
        pdir_split = pjoin(split, 'SubProcesses', 'P1_qq_qq')
        # Generation has to have COMPLETED for both, not merely made the
        # directory: a split the exporter cannot digest leaves the P directory
        # behind without its flavor tables, and every assertion below would
        # then fail on a missing file rather than on what it means to check.
        for pdir in (pdir_plain, pdir_split):
            self.assertTrue(os.path.isdir(pdir),
                            '%s was not generated' % pdir)
            self.assertTrue(
                os.path.isfile(pjoin(pdir, 'leshouche.inc')),
                '%s has no leshouche.inc -- the generation did not finish'
                % pdir)

        # (1) the peel really happened: an extra subprocess, and it is a ROUTER
        sub_plain = self._leshouche(pdir_plain)
        sub_split = self._leshouche(pdir_split)
        self.assertEqual(len(sub_split), len(sub_plain) + 1,
                         'the split did not add a subprocess to the group '
                         '(%d vs %d) -- MG_SPLIT_CROSSING did not fire'
                         % (len(sub_split), len(sub_plain)))

        # (2) and it PAYS: fewer compiled matrix elements than crossing-off
        n_plain, r_plain = self._counts(pdir_plain)
        n_split, r_split = self._counts(pdir_split)
        self.assertEqual(r_plain, 0,
                         '--use_crossing=False emitted %d router(s)' % r_plain)
        self.assertLess(n_split, n_plain,
                        'the split compiles %d matrix element(s), no better '
                        'than the %d of --use_crossing=False -- the peel costs '
                        'a subprocess and buys nothing' % (n_split, n_plain))
        self.assertEqual(r_split, len(sub_split) - n_split,
                         'every subprocess of the split group that is not a '
                         'compiled matrix element should be a router')

        # (3) the halves PARTITION the flavors. Both directions matter: a
        # combination covered twice is double counted, one covered by neither
        # is silently missing from the cross section.
        want = self._physical(pdir_plain)
        got = self._physical(pdir_split)
        self.assertEqual(
            sorted(got), sorted(want),
            'the split changed which physical flavor combinations the group '
            'covers (%d missing, %d new)'
            % (len(set(want) - set(got)), len(set(got) - set(want))))
        doubled = sorted(k for k, v in got.items() if v > 1)
        self.assertFalse(
            doubled,
            'the split covers %d flavor combination(s) TWICE, so they are '
            'double counted -- the two halves were re-identified into one '
            'pattern instead of staying complementary (e.g. %s)'
            % (len(doubled), doubled[:3]))

        # (4) the peeled sibling really is listed the OTHER way round -- that is
        # the whole reason it exists. Its rows are the flavour-changing
        # annihilation, and where the crossing-off build lists that class as
        # (q', q~') the sibling lists it as (q~', q'). Without this the test
        # would still pass if the peel produced a sibling identical to the
        # module it came from.
        peeled = sub_split[max(sub_split)]
        self.assertTrue(
            all(row[2] < 0 < row[3] for row in peeled),
            'the peeled subprocess does not list its final legs as '
            '(antiparticle, particle): %s' % (peeled[:3],))
        native = [row for rows in self._leshouche(pdir_plain).values()
                  for row in rows
                  if (tuple(sorted(row[:2])), tuple(sorted(row[2:])))
                  in set((tuple(sorted(r[:2])), tuple(sorted(r[2:])))
                         for r in peeled)]
        self.assertTrue(native, 'the crossing-off build has no counterpart for '
                                'the peeled class')
        self.assertTrue(
            all(row[3] < 0 < row[2] for row in native),
            'the crossing-off build already lists that class as '
            '(antiparticle, particle), so the peel swapped nothing: %s'
            % (native[:3],))

    def test_split_does_not_fire_for_an_exporter_that_cannot_take_it(self):
        """mg7 builds one module per leg pattern; handed a pattern split across
        two modules it raises "no valid flavor configurations found". The peel
        is a grouped-madevent optimisation and must stay off elsewhere."""
        outdir = self._generate('mg7', split=True, fmt='')
        self.assertTrue(
            os.path.isdir(outdir),
            'the default (mg7) export produced nothing with '
            'MG_SPLIT_CROSSING=on -- the split fired for a backend that '
            'cannot consume it')


class TestMadeventCrossingBaseColorFlow(unittest.TestCase):
    """A crossing BASE must pick the colour flow the same way with the crossing
    machinery on as with it off.

    Different code path from TestMadeventRouterColorSelection.  There is no
    router here: ``u u~ > g g`` is a cross-GROUP (Track B) dependent and simply
    reuses the compiled matrix element of ``g g > u u~``, which is the base.
    What the base has to get right is not a mask but its own recycled optim --
    and that is generated at RUN time by gen_ximprove, over the good-helicity
    set.  Keeping every helicity config there instead of the good-hel union
    looks harmless, because the |M|^2 sum is unchanged, but the same loop also
    accumulates AMP2 (the single-diagram multi-channel weights) and JAMP2 (the
    colour-flow weights), and a config whose |M|^2 vanishes still has non-zero
    individual diagrams and JAMPs.  For g g > q q~ that gave the s-channel
    config -- whose AMP2 is exactly zero over the good helicities -- about 10%
    of the subprocess, and SELECT_COLOR masks JAMP2 by ICONFIG, so those events
    took their flow from a polluted JAMP2 rather than the real one.

    Only the CORRELATION moves.  The cross section stayed right to 4 digits
    (the multi-channel weights are self-normalising), and so did the marginal
    helicity and the marginal colour distributions.  Seeing it needs the joint
    (helicity, colour) observable -- and for a class with two identical gluons
    the colour-only canonical form is not merely weak but structurally blind:
    it puts every event of g g > u u~ in ONE category, so its chi-square is
    identically 0 no matter what the code does.

    ``g g > u u~`` plus ``u u~ > g g`` rather than the dijet process the router
    test uses: same base/dependent crossing pair, but g g > g g is not there to
    take 81% of the events and starve the class being measured to 0.5%.
    """

    NEVENTS = 200000
    SEED = 777
    CLASS = ((21, 21), (-2, 2))    # g g > u u~
    # It takes roughly 10k events in the class to resolve the shift; the point
    # of this process is that essentially the whole sample lands there.
    MIN_CLASS = 20000

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='cross_base_col_')

    def tearDown(self):
        if os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _generate(self, options, name):
        from madgraph import MG5DIR
        outdir = pjoin(self.tmpdir, name)
        card = pjoin(self.tmpdir, 'cmd_%s.txt' % name)
        with open(card, 'w') as fsock:
            fsock.writelines(
                ['generate g g > u u~ %s\n' % options,
                 'add process u u~ > g g %s\n' % options,
                 'output madevent %s -f -nojpeg\n' % outdir,
                 'launch\n',
                 'set nevents %d\n' % self.NEVENTS,
                 'set iseed %d\n' % self.SEED,
                 # a broken local lhapdf kills the systematics step
                 'set use_syst False\n',
                 'set lpp2 -1\n'])
        subprocess.call([sys.executable, pjoin(MG5DIR, 'bin', 'mg5_aMC'), card])
        self.assertTrue(os.path.isdir(pjoin(outdir, 'SubProcesses')),
                        'madevent produced no output for %r' % (options or
                                                                'the default'))
        return outdir

    def test_crossing_base_colour_flow_matches_no_crossing(self):
        from madgraph.various import lhe_parser
        helper = TestMadeventRouterColorSelection

        crossed = self._generate('', 'on')
        plain = self._generate('--use_crossing=False', 'off')

        # The crossing really has to be in play, or this compares two identical
        # builds and passes on anything.
        base = pjoin(crossed, 'SubProcesses', 'P1_gg_qq',
                     'crossgroup_helunion.dat')
        self.assertTrue(
            os.path.exists(base),
            'the default build has no crossing base for g g > u u~ (no %s), so '
            'this test exercises no crossing at all' % os.path.basename(base))
        self.assertFalse(
            os.path.exists(pjoin(plain, 'SubProcesses', 'P1_gg_qq',
                                 'crossgroup_helunion.dat')),
            '--use_crossing=False still emitted a crossing base')

        ref = helper._topologies(plain, lhe_parser)
        got = helper._topologies(crossed, lhe_parser)
        self.assertIn(self.CLASS, ref,
                      'the --use_crossing=False build produced no %s events'
                      % helper._fmt(self.CLASS))
        self.assertIn(self.CLASS, got,
                      'the crossing build produced no %s events'
                      % helper._fmt(self.CLASS))

        rall, gall = ref[self.CLASS], got[self.CLASS]
        nref = sum(rall['colour'].values())
        ngot = sum(gall['colour'].values())
        self.assertGreaterEqual(
            min(nref, ngot), self.MIN_CLASS,
            '%s got %d/%d events, below the %d this comparison needs to '
            'resolve a colour-flow shift'
            % (helper._fmt(self.CLASS), nref, ngot, self.MIN_CLASS))

        # The colour-only form cannot see anything here -- assert that, so the
        # reason the joint form is required stays documented in the suite and a
        # future 'simplification' back to it fails loudly instead of quietly
        # testing nothing.
        self.assertEqual(
            len(set(list(rall['colour']) + list(gall['colour']))), 1,
            'the colour-only canonical form no longer merges the two flows of '
            '%s into one category; the blind spot this test exists for may '
            'have moved' % helper._fmt(self.CLASS))

        chi2, dof, crit = helper._homogeneity(rall['joint'], gall['joint'])
        logger.info('  %s: %d ref / %d crossed events, joint chi2 %.1f on %d '
                    'dof (critical %.1f)', helper._fmt(self.CLASS), nref, ngot,
                    chi2, dof, crit)
        self.assertGreater(dof, 1,
                           'the helicity-refined form separated only %d '
                           'category(ies), so it is no finer than the '
                           'colour-only one' % (dof + 1))
        self.assertLessEqual(
            chi2, crit,
            '%s: the (helicity, colour) mix differs between the crossing build '
            'and the --use_crossing=False build (chi2 = %.1f on %d dof, '
            'critical %.1f) -- the crossing base is not choosing the colour '
            'flow the module itself would'
            % (helper._fmt(self.CLASS), chi2, dof, crit))
