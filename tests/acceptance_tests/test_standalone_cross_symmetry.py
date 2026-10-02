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
"""Check the crossing-symmetry support of the fortran standalone output.

The standalone SMATRIX takes a flavor index (IFLAV / FLAV_IDX). Its range is
extended so that a single value carries both the flavor and a crossing to
apply, decoded as::

    K    = (IFLAV-1) / NFLAV               ! a row of the crossing table
    flav = mod(IFLAV-1, NFLAV) + 1         ! the index used for masking/...

Row K of the table generated for the matrix element is a slot permutation D:
input slot k of the crossed call (the crossed process' own leg order) is fed
to base slot D[k], the leg charge conjugated when it changes side. Row 0 is the
identity, so IFLAV in [1,NFLAV] keeps its meaning and existing callers are
unaffected. A folding output holds the rows of its recorded crossings; the
bare single-process outputs of these tests are written with
``--crossing_table=all``, which adds every ordered choice of initial legs (the
leg sent to the final state taking the slot of the leg replacing it), and the
tests look the row of a crossing up by its permutation D through
GET_CROSS_PINV -- no test knows how the rows are numbered.

Moving a particle across the initial/final state flips its NSF/NSV helas flag
(which is what negates the momentum stored in the wavefunction), so the
crossed call evaluates the same analytic amplitude in a different kinematic
region.

The processes u u~ > g g and u g > u g are exactly each other's crossing under
D = (0, 2, 1, 3): slot 2 of u g > u g takes the outgoing gluon of slot 3, now
incoming, and slot 3 the incoming u~ of slot 2, now an outgoing u. Because the
crossing also reorders the legs, the crossed call takes the *other* process's
natural momentum layout, so this test feeds both codes the very same momenta.

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
import madgraph.various.misc as misc

pjoin = os.path.join

# The two processes are each other's crossing under D_2_3.
PROC_QQ_GG = 'u u~ > g g'
PROC_QG_QG = 'u g > u g'
# A crossed partner of `g g > q q~` (base slots 2 and 3 swapped across), used
# by the madmatrix tests that need the crossing to be a RECORDED one.
PROC_GQX_GQX = 'g u~ > g u~'

# A CHIRAL pair: the W+ couples only to a left-handed u and a right-handed d~, so
# every external quark is 100% polarized and the per-leg density matrix diagonal
# is fully asymmetric ((++) empty, (--) full, or vice versa). That is what makes
# a crossed-fermion helicity FLIP detectable: on u u~ > g g the fermion density
# is (++)==(--), so a flip would be invisible; here it would swap a full entry
# with an empty one. u d~ > w+ g is mapped onto u g > w+ d by D_2_LAST:
# the incoming d~ becomes the outgoing d of the last slot (the crossed, still
# 100%-polarized fermion), the outgoing g becomes incoming.
PROC_UDX_WPG = 'u d~ > w+ g'
PROC_UG_WPD = 'u g > w+ d'

# q q~ > g q q~ is likewise mapped onto q g > q q q~ by the same slot 2 <-> 3
# crossing (D_2_3_5): the incoming q~ becomes the outgoing q of slot 3 and the outgoing g
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
# ... and its crossing partner under the same constraint, so a whole merged
# split-orders table can be swept through the crossing (see
# test_split_orders_merged_flavor_crossing_every_flavor).
PROC_MERGED_QQX_GQQX_SO = '_quark _anti_quark > g _quark _anti_quark QED^2==0'

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
    'GET_CROSS_PINV', 'GET_SPINCOL_CROSS', 'GET_IDENT_CROSS', 'CROSS_GHIDX',
    'XPERM', 'XPINV', 'XSPINCOL', 'GHFILT']

# The crossings the tests probe, as the permutation D of a crossing-table row
# (0-based): input slot k of the crossed call is fed to base slot D[k].
# Particle 2 swapped with particle 3 (the former code I=0, J=3), on a 2->2 and
# on a 2->3.
D_2_3 = (0, 2, 1, 3)
D_2_3_5 = (0, 2, 1, 3, 4)
# Particle 2 swapped with the LAST particle (the former I=0, J=NEXTERNAL).
D_2_LAST = (0, 3, 2, 1)
# A genuine 3-cycle, which no (I,J) code could name: on u u~ > g g, leg 1 of
# the crossed call takes the u~ (still incoming), leg 2 a gluon (now incoming)
# and leg 3 the u (now an outgoing u~), i.e. u~ g > u~ g. Not an involution (D != D^-1), so a
# consumer reading the permutation in the wrong direction cannot hide.
D_3CYCLE = (1, 2, 0, 3)
IFLAV_IDENTITY = 1


def _iflav(row, flav, nflav):
    """Encode a crossing-table row and a flavor index into the extended
    IFLAV."""
    return row * nflav + flav


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


# Subprocess probe for the good-helicity remap relation. Run against a compiled
# matrix2py module: for every row of the crossing table (read back through
# py_get_crossing, the f2py face of GET_CROSS_PINV), the crossed good-helicity
# set -- the rows where py_smatrixhel_idx is non-zero, unioned over many
# phase-space points -- must equal the identity good-helicity set mapped
# through tau, the sign flip of the legs that change side, indexed by BASE slot
# (config h -> (SB[b]*nhel[b,h])_b). This is the invariant the generated
# CROSS_GHIDX / GHFILT encode, so a wrong table, or signs read in the input-slot
# view (SD) instead of the base view (SB) -- identical for an involution,
# different for the 3-cycles the table now holds -- breaks it. Run in a
# subprocess: importing an f2py .so into the test interpreter would leak a
# compiled module and clash across tests.
#
# GOTCHA locked in by this probe: 3 phase-space points are NOT enough -- for
# u u~ > g g, the former cross=23 then showed 6 non-zero rows instead of 8 (an
# accidental zero at the probed points). NPTS is deliberately >= 12.
_GOODHEL_PROBE = r'''
import sys, math
import numpy as np
sys.path.insert(0, %(pdir)r)
import matrix2py as m

NINITIAL = %(ninitial)d
NPTS = %(npts)d

def crossing_row(flav_idx, nexternal):
    """(D, SB) of the table row an extended index selects, None if it names
    no crossing. D[k] = base slot input slot k is fed to; SB[b] = -1 when base
    leg b changes side."""
    pinv, sgni, flav_out = m.py_get_crossing(flav_idx)
    if int(flav_out) == 0:
        return None
    D = [int(x) - 1 for x in pinv]
    SD = [int(x) for x in sgni]
    B = [0] * nexternal
    for k, b in enumerate(D):
        B[b] = k
    return D, [SD[B[b]] for b in range(nexternal)]

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

# per base leg, its helicity states in code order (the table is the
# mixed-radix product, so a leg's values appear in state order)
leg_states = [[] for _ in range(nexternal)]
for h in range(ncomb):
    for b in range(nexternal):
        if int(nhel[b, h]) not in leg_states[b]:
            leg_states[b].append(int(nhel[b, h]))

def base_row_of_code(code, D):
    """SMATRIXHEL takes the crossed process's OWN helicity code at an
    extended index (CROSS_HELCODE): crossed leg k carries base leg D[k] with
    its states, so decode over those and put each value in its base slot --
    the base row the crossing evaluates."""
    r, cfg = code - 1, [0] * nexternal
    for k in reversed(range(nexternal)):
        n = len(leg_states[D[k]])
        cfg[D[k]] = leg_states[D[k]][r %% n]
        r //= n
    return row_of[tuple(cfg)]

def good_set(flav_idx, D=None):
    """The good BASE rows of an extended index (D: its crossing row)."""
    good = set()
    for P in ps:
        for h in range(1, ncomb + 1):
            if abs(m.py_smatrixhel_idx(P, h, flav_idx)) > 1e-30:
                good.add(h if D is None else base_row_of_code(h, D))
    return good

g_id = good_set(1)
assert g_id, 'identity has no good helicity -- probe is broken'
checked = genuine = cycles = 0
for K in range(1, ncross):
    flav_idx = K * nflav + 1
    row = crossing_row(flav_idx, nexternal)
    assert row is not None, 'row %%d names no crossing' %% K
    D, SB = row
    # Every row is a crossing some caller asked for: none may be null.
    tot = sum(abs(m.py_smatrixhel_idx(ps[0], h, flav_idx))
              for h in range(1, ncomb + 1))
    assert tot > 0, 'row %%d (D=%%s) evaluates to zero' %% (K, D)
    # TAU, not a helicity-row permutation: the matrix element evaluates base
    # slot b at NHEL(b)*SB(b) with the base slot order (APPLY_CROSSING_TABLE
    # permutes only the momenta), so the map relating a crossed row to its
    # identity row is SB[b]*nhel[b,h] -- no D[] indexing. This is the same map
    # the recycled optim and the madmatrix lanes realise, which is the point:
    # one good-helicity relation describes every backend. tau is a clean
    # bijection (each leg's states are closed under negation), so this stays an
    # EQUALITY rather than a containment.
    tau = {}
    for h in range(ncomb):
        cfg = tuple(SB[b] * nhel[b, h] for b in range(nexternal))
        hp = row_of.get(cfg)
        assert hp is not None, 'row %%d: tau is not a row bijection' %% K
        tau[h + 1] = hp
    expected = {tau[h] for h in g_id}
    g_cr = good_set(flav_idx, D)
    assert g_cr == expected, (
        'row %%d (D=%%s): crossed good-hel %%s != tau(identity) %%s'
        %% (K, D, sorted(g_cr), sorted(expected)))
    checked += 1
    if -1 in SB:
        genuine += 1
    if any(D[D[k]] != k for k in range(nexternal)):
        cycles += 1
assert checked == ncross - 1, 'checked %%d of %%d rows' %% (checked, ncross - 1)
assert genuine >= 1, 'no genuine (side-changing) crossing was checked'
assert cycles >= 1, 'no non-involution row was checked'
print('GHREMAP_RELATION_OK checked=%%d genuine=%%d cycles=%%d points=%%d' %%
      (checked, genuine, cycles, NPTS))
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



def _pin_crossing(options, on=True):
    """Return `options` with the crossing choice stated explicitly.

    This suite is *about* crossing, so nothing in it may lean on the shipped
    default (on) in either direction: a caller that already passed
    --use_crossing=... keeps its choice, everyone else gets it pinned here.
    Flipping the product default must never silently turn one of these tests
    into a test of the other mode; TestCrossingProductDefault is the one place
    that reads the default itself.
    """
    if '--use_crossing' in options:
        return options
    return ('%s --use_crossing=%s' % (options, on)).strip()

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

        A single process records no crossing, so its crossing table would hold
        the identity alone: the output asks for every applicable crossing
        (--crossing_table=all), the rows these tests probe.
        """
        outdir = pjoin(self.tmpdir, name)
        self.cmd.exec_cmd('set automatic_html_opening False')
        self.cmd.exec_cmd('set group_subprocesses False')
        self.cmd.exec_cmd('set apply_flavor_grouping True')
        self.cmd.exec_cmd('import model sm')
        self.cmd.exec_cmd(
            ('generate %s %s' % (process, _pin_crossing(options))).strip())
        self.cmd.exec_cmd('output standalone_fortran %s -f --crossing_table=all'
                          % outdir)

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
        # GET_CROSS_PINV only exists with the crossing machinery; a driver
        # built against a matrix.f without it must not reference it.
        if re.search(r'SUBROUTINE\s+GET_CROSS_PINV\b', self._matrix_code(pdir)):
            rows_call = '''         READ(42,*) NFL, NCR
         DO K=0,NCR-1
            CALL GET_CROSS_PINV(K*NFL+1, PINV, SGNI, XIDX)
            WRITE(*,*) 'ROW= ', K, XIDX, (PINV(I),I=1,NEXTERNAL)
         ENDDO'''
        else:
            rows_call = "         WRITE(*,*) 'NOROWS'"
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
      INTEGER PINV(NEXTERNAL), SGNI(NEXTERNAL), XIDX, NFL, NCR, K
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
      ELSEIF (MODE.EQ.6) THEN
C        The crossing table, one row per line: K, the base flavor the index
C        K*NFLAV+1 resolves to (0: the row names no crossing) and D (1-based),
C        the base slot each input slot is fed to (GET_CROSS_PINV).
%(rows_call)s
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
                                  'nhel_idx_call': nhel_idx_call,
                                  'rows_call': rows_call})

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

    def _read_ncross(self, pdir):
        """NCROSS of a generated process: the rows of its crossing table."""
        with open(pjoin(pdir, 'matrix.f')) as fsock:
            match = re.search(r'PARAMETER\s*\(NCROSS=(\d+)\)', fsock.read())
        self.assertTrue(match, 'Could not read NCROSS from %s' % pdir)
        return int(match.group(1))

    def _crossing_rows(self, pdir):
        """{D: K} over the valid rows of the crossing table, D the 0-based
        permutation (input slot k fed to base slot D[k]), read back from the
        compiled GET_CROSS_PINV."""
        output = self._probe(pdir, ['6', '%d %d' % (self._read_nflav(pdir),
                                                    self._read_ncross(pdir))])
        rows = {}
        for line in re.findall(r'ROW=\s*(.*)', output):
            fields = [int(token) for token in line.split()]
            if fields[1] == 0:
                continue
            rows[tuple(d - 1 for d in fields[2:])] = fields[0]
        self.assertTrue(rows, 'No crossing-table row from %s, got:\n%s'
                        % (pdir, output))
        return rows

    def _crossing_rows_f2py(self, pdir):
        """The same {D: K} map through the compiled f2py module
        (FlavorDispatch.crossing_for_index / PY_GET_CROSSING), for a directory
        with no fortran driver. Requires the module built (_build_f2py)."""
        script = '''
import sys
sys.path.insert(0, %(pdir)r)
import matrix2py
from flavor_dispatch import FlavorDispatch
me = FlavorDispatch(matrix2py)
nflav, _nexternal, ncross = me.flavor_layout()
for K in range(ncross):
    row = me.crossing_for_index(K * nflav + 1)
    if row is not None:
        print('ROW= %%d %%s' %% (K, ' '.join(str(d) for d in row[0])))
''' % {'pdir': pdir}
        script_path = pjoin(pdir, 'rows_probe.py')
        with open(script_path, 'w') as fsock:
            fsock.write(script)
        output = subprocess.Popen(
            [sys.executable, script_path], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, cwd=pdir).communicate()[0].decode()
        rows = {}
        for line in re.findall(r'ROW=\s*(.*)', output):
            fields = [int(token) for token in line.split()]
            rows[tuple(fields[1:])] = fields[0]
        self.assertTrue(rows, 'No crossing-table row from the f2py module in '
                        '%s, got:\n%s' % (pdir, output))
        return rows

    def _row(self, pdir, perm, rows=None):
        """Row K of the crossing table holding the permutation `perm` (D,
        0-based); fails when the table has no such row."""
        rows = self._crossing_rows(pdir) if rows is None else rows
        self.assertIn(tuple(perm), rows,
                      'The crossing table of %s has no row D=%s (rows: %s)'
                      % (pdir, perm, sorted(rows)))
        return rows[tuple(perm)]

    def _cross_iflav(self, pdir, perm, flav=1, nflav=None, rows=None):
        """The extended IFLAV evaluating base flavor `flav` through the row
        holding the permutation `perm`."""
        if nflav is None:
            nflav = self._read_nflav(pdir)
        return _iflav(self._row(pdir, perm, rows), flav, nflav)

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

        IFLAV = K*NFLAV + flav, so a crossing-table row cannot be turned into
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
        """Row 0 must be the identity and leave the behaviour untouched."""
        qq_gg = self._generate(PROC_QQ_GG, 'Proc_qq_gg')
        self.assertEqual(self._row(qq_gg, (0, 1, 2, 3)), 0,
                         'Row 0 of the crossing table is not the identity')
        momenta = self._phase_space(self.cos_thetas[0])
        plain = self._run(qq_gg, momenta, IFLAV_IDENTITY)
        self.assertNotEqual(plain, 0.0,
                            'Sanity check failed: %s gives a null matrix element'
                            % PROC_QQ_GG)
        # An index past the last row names no crossing: SMATRIX returns 0
        # rather than evaluating some other row (or reading past the tables).
        self.assertEqual(
            self._run(qq_gg, momenta,
                      _iflav(self._read_ncross(qq_gg), 1, nflav=1)), 0.0,
            'an index past the crossing table does not give a zero ME')

    def test_qq_gg_crossed_gives_qg_qg(self):
        """u u~ > g g with particle 2 <-> 3 crossed must give u g > u g."""
        qq_gg = self._generate(PROC_QQ_GG, 'Proc_qq_gg')
        qg_qg = self._generate(PROC_QG_QG, 'Proc_qg_qg')
        self._assert_crossing(
            crossed_dir=qq_gg, crossed_iflav=self._cross_iflav(qq_gg, D_2_3),
            reference_dir=qg_qg, label='%s crossed (D=%s) vs %s'
            % (PROC_QQ_GG, D_2_3, PROC_QG_QG))

    def test_qq_gg_crossed_with_last_particle(self):
        """Particle 2 must be crossable with the last particle.

        (The former I*(NEXTERNAL+1)+J code needed its NEXTERNAL+1 base for
        this one.) Swapping particle 2 with particle 4 in u u~ > g g turns the
        incoming u~ into an outgoing u sitting in slot 4 and the outgoing g of
        slot 4 into an incoming one, so the legs come out ordered as
        u g > g u: the same physics as u g > u g with the two final legs
        exchanged.
        """
        qq_gg = self._generate(PROC_QQ_GG, 'Proc_qq_gg')
        qg_qg = self._generate(PROC_QG_QG, 'Proc_qg_qg')
        self._assert_crossing(
            crossed_dir=qq_gg, crossed_iflav=self._cross_iflav(qq_gg, D_2_LAST),
            reference_dir=qg_qg, reference_perm=(0, 1, 3, 2),
            label='%s crossed (D=%s) vs %s with final legs swapped'
            % (PROC_QQ_GG, D_2_LAST, PROC_QG_QG))

    def test_qq_gg_three_cycle(self):
        """3-cycle rows: u u~ > g g evaluating u~ g > u~ g, u g > u g
        evaluating g u~ > u~ g.

        No (I,J) code could name them: D = (1, 2, 0, 3) sends leg 1 to base
        slot 2, leg 2 to base slot 3 and leg 3 to base slot 1. D is not an
        involution (D != D^-1), so a consumer reading the permutation in the
        wrong direction -- the momentum gather, the crossed PDG, the NSF flags
        or the denominator -- gives a different number instead of hiding
        behind a symmetric swap. On u g > u g the two directions also differ
        in the crossed initial state (g u~, spin*colour 96, against u~ u, 36),
        which pins the denominator; the crossed PDG and denominator the f2py
        accessors report are compared too.

        The compiled rows must also be the python table's, row for row: rows
        are looked up here through the compiled GET_CROSS_PINV, and the table
        of every applicable crossing holds D^-1 too, so an emitter swapping
        the two views consistently would otherwise pass unseen.
        """
        import madgraph.iolibs.crossing_table as crossing_table
        cases = [(PROC_QQ_GG, 'Proc_qq_gg', 'u~ g > u~ g', 'Proc_qxg_qxg',
                  (-2, 21, -2, 21)),
                 (PROC_QG_QG, 'Proc_qg_qg', 'g u~ > u~ g', 'Proc_gqx_qxg',
                  (21, -2, -2, 21))]
        expected_rows = dict((D, K) for K, D in
                             enumerate(crossing_table.applicable_perms(4, 2)))
        for base_line, base_name, ref_line, ref_name, pdgs in cases:
            with self.subTest(base=base_line):
                base = self._generate(base_line, base_name)
                reference = self._generate(ref_line, ref_name)
                self.assertEqual(self._crossing_rows(base), expected_rows,
                                 'the compiled crossing table of %s is not the '
                                 'python one' % base_line)
                crossed_iflav = self._cross_iflav(base, D_3CYCLE)
                iden, got = self._nhel_idx(base, crossed_iflav)
                self.assertEqual(got, pdgs,
                                 'the 3-cycle row of %s should evaluate %s, '
                                 'got %s' % (base_line, ref_line, got))
                self.assertEqual(iden, self._nhel_idx(reference,
                                                      IFLAV_IDENTITY)[0],
                                 'crossed denominator differs from %s'
                                 % ref_line)
                self._assert_crossing(
                    crossed_dir=base, crossed_iflav=crossed_iflav,
                    reference_dir=reference, label='%s crossed (D=%s) vs %s'
                    % (base_line, D_3CYCLE, ref_line))

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
                                    self._cross_iflav(qqx_gqqx, D_2_3_5, 1,
                                                      nflav=nflav))
                reference = self._run(qg_qqqx, momenta, IFLAV_IDENTITY)

                self.assertNotEqual(
                    reference, 0.0,
                    'Sanity check failed: %s gives a null matrix element'
                    % (PROC_QG_QQQX % {'q': quark}))
                scale = max(abs(crossed), abs(reference), 1e-99)
                self.assertLessEqual(
                    abs(crossed - reference) / scale, self.tolerance,
                    '%s crossed (D=%s) disagrees with %s: '
                    'crossed=%r reference=%r'
                    % (PROC_QQX_GQQX % {'q': quark}, D_2_3_5,
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
        row_2_3 = self._row(merged_a, D_2_3_5)

        unmapped = []
        for flav in range(1, nflav_a + 1):
            positions = self._flavor_positions(merged_a, flav)
            # Caller slot 2 holds leg 3 (the gluon) and slot 3 holds leg 2.
            crossed = tuple(positions[d] for d in D_2_3_5)
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
                                          _iflav(row_2_3, flav, nflav=nflav_a))
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
        partner is the missing degree of freedom. Particle 2 swapped with
        particle 3 or with particle 4 crosses one or the other of the two final
        quarks, and those land on different flavors of the target. The two
        coincide only when the two final quarks already share a flavor, so the
        count works out exactly:

            16 flavors x 2 crossings - 4 degenerate = 28

        The swap with particle 4 leaves the legs ordered (q, q~, q, g, q~)
        instead of the target's (q, q~, g, q, q~), hence the momentum swap of
        slots 3 and 4.
        """
        merged_a = self._generate(PROC_MERGED_QQX_GQQX, 'Proc_merged_a')
        merged_b = self._generate(PROC_MERGED_QG_QQQX, 'Proc_merged_b')
        nflav_a = self._read_nflav(merged_a)
        nflav_b = self._read_nflav(merged_b)
        momenta = self._phase_space_2to3()
        rows_b = self._crossing_rows(merged_b)
        # particle 2 swapped with particle 4
        d_2_4 = (0, 3, 2, 1, 4)

        covered = {}
        for flav_b in range(1, nflav_b + 1):
            positions = self._flavor_positions(merged_b, flav_b)
            variants = (
                # 2 <-> 3: legs already come out in the target's order.
                (D_2_3_5, (positions[0], positions[2], positions[1],
                           positions[3], positions[4]), None),
                # 2 <-> 4: cross the other final quark, then reorder slots 3/4.
                (d_2_4, (positions[0], positions[3], positions[1],
                         positions[2], positions[4]), (0, 1, 3, 2, 4)),
            )
            for cross_perm, target_positions, perm in variants:
                flav_a = self._flavor_index(merged_a, target_positions)
                self.assertGreaterEqual(
                    flav_a, 1,
                    'Crossed flavor %s (from %s flavor %s, D=%s) has no '
                    'counterpart in %s'
                    % (target_positions, PROC_MERGED_QG_QQQX, flav_b, cross_perm,
                       PROC_MERGED_QQX_GQQX))
                covered.setdefault(flav_a, []).append((flav_b, cross_perm))

                with self.subTest(flav_b=flav_b, cross_perm=cross_perm):
                    crossed_momenta = momenta if perm is None else \
                        [momenta[index] for index in perm]
                    crossed_value = self._run(merged_b, crossed_momenta,
                                              self._cross_iflav(
                                                  merged_b, cross_perm, flav_b,
                                                  nflav=nflav_b, rows=rows_b))
                    reference = self._run(merged_a, momenta, flav_a)
                    scale = max(abs(crossed_value), abs(reference), 1e-99)
                    self.assertLessEqual(
                        abs(crossed_value - reference) / scale, self.tolerance,
                        '%s flavor %s crossed (D=%s) disagrees with %s flavor '
                        '%s: crossed=%r reference=%r'
                        % (PROC_MERGED_QG_QQQX, flav_b, cross_perm,
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
                                        self._cross_iflav(qq_gg, D_2_3), leg=2)
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

        u d~ > w+ g crossed by D_2_LAST is u g > w+ d; its outgoing d
        is the incoming d~ that swapped sides, still 100% polarized by the W. The
        density matrix is per helicity, so it is the probe that pins how that
        crossed leg's helicity is LABELLED -- the same no-flip convention the
        madevent cross-group event helicity (DSIG_XGHEL) depends on. Every leg,
        crossed vs natively generated, must agree term by term.
        """
        udx_wpg = self._generate(PROC_UDX_WPG, 'Proc_udx_wpg')
        ug_wpd = self._generate(PROC_UG_WPD, 'Proc_ug_wpd')
        crossed_iflav = self._cross_iflav(udx_wpg, D_2_LAST)
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
        crossed_iflav = _iflav(
            self._row(udx_wpg, D_2_LAST, self._crossing_rows_f2py(udx_wpg)),
            1, nflav=1)
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

    def test_split_orders_merged_flavor_crossing_every_flavor(self):
        """The merged-flavor crossing sweep, on the split-orders template.

        The twin of test_merged_flavor_crossing_every_flavor, and here for the
        same reason as test_split_orders_density_diagonal_matches_smatrix:
        matrix_standalone_splitOrders_v4.inc is a SEPARATE template with its own
        SMATRIX, so making the default one cross correctly says nothing about
        it. It carried no crossing machinery at all until
        fill_crossing_replace_dict_so, while the generator happily folded the
        crossed subprocesses onto their base -- 50 of the 65 flavor columns of
        `p p > j j QCD^2==4` had no entry point left, and the extended FLAV_IDX
        that names them returned 0 in silence.

        Sweeping the whole merged table is what makes this bite: the crossed
        denominator is rebuilt per flavor (GET_SPINCOL_CROSS *
        GET_IDENT_CROSS), so a denominator taken from the representative flavor
        instead of the actual one shows up as a clean factor 2 on the rows
        where the two final quarks differ.
        """
        merged_a = self._generate(PROC_MERGED_QQX_GQQX_SO, 'Proc_so_cross_a',
                                  split_orders=True)
        merged_b = self._generate(PROC_MERGED_QG_QQQX_SO, 'Proc_so_cross_b',
                                  split_orders=True)
        # Guard the premise twice over: the split-orders template, and the
        # crossing machinery actually written into it.
        code_a = self._matrix_code(merged_a)
        self.assertIn('SMATRIX_SPLITORDERS', code_a,
                      'Expected %s to use the split-orders template; this test '
                      'would otherwise just retest the default one'
                      % PROC_MERGED_QQX_GQQX_SO)
        self.assertIn('GET_CROSS_PERM', code_a,
                      'The split-orders matrix.f carries no crossing '
                      'machinery, so an extended FLAV_IDX cannot be decoded '
                      'and every assertion below would compare zeros')
        nflav_a = self._read_nflav(merged_a)
        self.assertGreater(nflav_a, 1,
                           'Expected a merged multi-flavor matrix element, got '
                           'NFLAV=%s' % nflav_a)
        momenta = self._phase_space_2to3()
        row_2_3 = self._row(merged_a, D_2_3_5)

        unmapped = []
        checked = 0
        for flav in range(1, nflav_a + 1):
            positions = self._flavor_positions(merged_a, flav)
            # Caller slot 2 holds leg 3 (the gluon) and slot 3 holds leg 2.
            crossed = tuple(positions[d] for d in D_2_3_5)
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
                                          _iflav(row_2_3, flav, nflav=nflav_a))
                reference_momenta = momenta if reference_perm is None else \
                    [momenta[index] for index in reference_perm]
                reference = self._run(merged_b, reference_momenta, target)
                self.assertNotEqual(
                    crossed_value, 0.0,
                    'Crossed flavor %s evaluated to exactly zero, which is what '
                    'a matrix element with no crossing decoder returns for an '
                    'extended FLAV_IDX' % flav)
                scale = max(abs(crossed_value), abs(reference), 1e-99)
                self.assertLessEqual(
                    abs(crossed_value - reference) / scale, self.tolerance,
                    'split-orders flavor %s (positions %s) crossed disagrees: '
                    'crossed=%r reference=%r (ratio %r)'
                    % (flav, positions, crossed_value, reference,
                       reference / crossed_value if crossed_value else None))
                checked += 1

        self.assertFalse(unmapped,
                         'Crossed flavors with no counterpart in %s: %s'
                         % (PROC_MERGED_QG_QQQX_SO, unmapped))
        self.assertGreater(checked, 1,
                           'Only %s flavor compared; the sweep is the point'
                           % checked)

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
        For u u~ > g g the identity (IFLAV=1) is itself, and the D_2_3
        crossing is u g > u g: leg 2's u~ (pdg -2) becomes an outgoing u
        (pdg +2) in slot 3, and IDEN goes 72 -> 96.
        """
        qq_gg = self._generate(PROC_QQ_GG, 'Proc_qq_gg')

        iden_id, pdg_id = self._nhel_idx(qq_gg, IFLAV_IDENTITY)
        self.assertEqual(iden_id, 72,
                         'Identity IDEN wrong: %s' % iden_id)
        self.assertEqual(pdg_id, (2, -2, 21, 21),
                         'Identity PDG wrong: %s' % (pdg_id,))

        iden_cr, pdg_cr = self._nhel_idx(qq_gg, self._cross_iflav(qq_gg, D_2_3))
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
        the D_2_3 crossing is u g > u g. Skips if f2py cannot build here.
        """
        pdir = self._output_standalone(PROC_QQ_GG, 'Proc_qq_gg_f2py')
        self._build_f2py(pdir)
        rows = self._crossing_rows_f2py(pdir)
        # every ordered choice of the two initial legs among four
        self.assertEqual(len(rows), 12, sorted(rows))
        iflav_2_3 = _iflav(self._row(pdir, D_2_3, rows), 1, nflav=1)

        # Run in a subprocess: importing an f2py .so into the test interpreter
        # would leak a compiled module and clash across tests.
        script = '''
import sys, math, numpy as np
sys.path.insert(0, %(pdir)r)
import matrix2py
from flavor_dispatch import FlavorDispatch
me = FlavorDispatch(matrix2py)
me.initialisemodel(%(card)r)
IDX = %(iflav)d
assert me.flavor_layout() == (1, 4, 12), me.flavor_layout()
assert me.pdg_for_index(1) == (2, -2, 21, 21), me.pdg_for_index(1)
assert me.pdg_for_index(IDX) == (2, 21, 2, 21), me.pdg_for_index(IDX)
assert me.crossing_for_index(IDX)[0] == %(perm)r, me.crossing_for_index(IDX)
assert me.crossing_for_index(IDX)[1] == (1, -1, -1, 1)
assert me.crossing_for_index(13) is None      # past the last row
assert me.find_pdg([2, -2, 21, 21]) == 1
assert me.find_pdg([2, 21, 2, 21]) == IDX
assert me.find_pdg([6, -6, 21, 21]) is None   # unreachable process
E = 500.0; c = 0.3; s = math.sqrt(1.0 - c * c)
P = np.asfortranarray(np.array([[E, 0, 0, E], [E, 0, 0, -E],
    [E, E * s, 0, E * c], [E, -E * s, 0, -E * c]]).T)
direct = me.smatrix(P, IDX)
via = me.matrix_element_pdg(P, [2, 21, 2, 21])
assert abs(direct - via) <= 1e-11 * abs(direct), (direct, via)
assert direct > 0.0
print("F2PY_PDG_OK")
''' % {'pdir': pdir, 'iflav': iflav_2_3, 'perm': D_2_3,
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

    def test_all_matrix_pdg_dispatch_reaches_folded_crossings(self):
        """The combined f2py module (--prefix=int) selects a process by its
        PDG codes (smatrixhel). A crossed subprocess folded onto its base has
        no PDG branch of its own, and used to come back as 0 with no error --
        u d~ > w+ g of p p > w+ j, now that crossing is on by default. It must
        give what the --use_crossing=False output gives."""
        outs = {}
        for name, options in (('on', ''), ('off', ' --use_crossing=False')):
            out = pjoin(self.tmpdir, 'all_matrix_' + name)
            self.cmd.exec_cmd('set automatic_html_opening False')
            self.cmd.exec_cmd('import model sm')
            self.cmd.exec_cmd('generate p p > w+ j --use_crossing=True')
            self.cmd.exec_cmd('output standalone_fortran %s -f --prefix=int%s'
                              % (out, options))
            sub = pjoin(out, 'SubProcesses')
            with open(os.devnull, 'w') as devnull:
                retcode = subprocess.call(['make', 'f2py'], cwd=sub,
                                          stdout=devnull, stderr=devnull)
            if retcode != 0 or not [n for n in os.listdir(sub)
                                    if n.startswith('all_matrix2py')]:
                raise unittest.SkipTest('could not build all_matrix2py')
            outs[name] = out
        self.assertEqual(len([d for d in os.listdir(pjoin(outs['on'],
                              'SubProcesses')) if d.startswith('P')]), 1,
                         'p p > w+ j folded nothing')
        script = '''
import sys, os, math, numpy as np
sub = os.path.join(sys.argv[1], 'SubProcesses')
sys.path.insert(0, sub)
import all_matrix2py as m
m.initialise(os.path.join(sys.argv[1], 'Cards', 'param_card.dat'))
E, mw, c = 500.0, 80.419, 0.3
e3 = (4 * E * E + mw * mw) / (4 * E); q = math.sqrt(e3 * e3 - mw * mw)
s = math.sqrt(1 - c * c)
P = np.array([[E, 0, 0, E], [E, 0, 0, -E], [e3, q * s, 0, q * c],
              [2 * E - e3, -q * s, 0, -q * c]]).T
for pdgs in ([21, 2, 24, 1], [2, -1, 24, 21], [21, -1, 24, -2]):
    print('ME %r' % m.smatrixhel(pdgs, -1, P, 0.118, 0, -1))
'''
        values = {}
        for name, out in outs.items():
            path = pjoin(self.tmpdir, 'probe_%s.py' % name)
            with open(path, 'w') as fsock:
                fsock.write(script)
            output = subprocess.Popen(
                [sys.executable, path, out], stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT).communicate()[0].decode()
            values[name] = [float(v) for v in re.findall(r'^ME (\S+)$',
                                                          output, re.M)]
            self.assertEqual(len(values[name]), 3, output)
        for on, off in zip(values['on'], values['off']):
            self.assertGreater(off, 0.0)
            self.assertLessEqual(abs(on - off), 1e-10 * off,
                                 (values['on'], values['off']))

    def _assert_goodhel_relation(self, process, name, ninitial, npts=16):
        """Compiled-module check of the GHREMAP good-helicity relation.

        Builds the f2py module for `process` and, for every row of its
        crossing table, asserts the crossed good-helicity set equals the
        identity's mapped through TAU, the sign-only map every backend can
        realise (the invariant the shared good-helicity filter encodes).
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
        mapped through tau, for every row (>=12 points, so the accidental-zero
        undercount seen on the former cross=23 cannot mask a bug)."""
        self._assert_goodhel_relation(PROC_QQ_GG, 'Proc_qq_gg_goodhel',
                                      ninitial=2)

    def test_goodhel_relation_qq_ggg(self):
        """Same relation on a 2->3 (u u~ > g g g): more crossings, beam swaps
        and 3-cycles included."""
        self._assert_goodhel_relation('u u~ > g g g', 'Proc_qq_ggg_goodhel',
                                      ninitial=2)

    def test_qg_qg_crossed_gives_qq_gg(self):
        """u g > u g with particle 2 <-> 3 crossed must give u u~ > g g."""
        qq_gg = self._generate(PROC_QQ_GG, 'Proc_qq_gg')
        qg_qg = self._generate(PROC_QG_QG, 'Proc_qg_qg')
        self._assert_crossing(
            crossed_dir=qg_qg, crossed_iflav=self._cross_iflav(qg_qg, D_2_3),
            reference_dir=qq_gg, label='%s crossed (D=%s) vs %s'
            % (PROC_QG_QG, D_2_3, PROC_QQ_GG))

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

    def _assert_decay_crossing(self, base_dir, base_line, ref_line, perm,
                               pdgs):
        """The base decay-chain SMATRIX at a crossing must reproduce a
        fully-generated (--use_crossing=False) build of the crossed decay chain.

        `perm` is the crossing-table row (D, over the leaves) and `pdgs` the
        crossed leaf signature, the order the momenta must be supplied in; it
        is both the reference process order and the momentum order fed to both
        builds. The base carries the crossing through the extended IFLAV, the
        reference evaluates it as its own identity -- the two must agree to
        machine precision.
        """
        ref_dir = self._generate(ref_line, 'Proc_dc_ref_%s'
                                 % ''.join(str(d) for d in perm),
                                 options='--use_crossing=False')
        crossed_iflav = self._cross_iflav(base_dir, perm)
        self.assertEqual(self._nhel_idx(base_dir, crossed_iflav)[1],
                         tuple(pdgs),
                         'Row D=%s of %s does not evaluate %s'
                         % (perm, base_line, ref_line))
        momenta = self._massive_2ton(ref_dir, pdgs)
        crossed = self._run(base_dir, momenta, crossed_iflav)
        reference = self._run(ref_dir, momenta, IFLAV_IDENTITY)
        self.assertNotEqual(reference, 0.0,
                            'Sanity check failed: %s gives a null matrix element'
                            % ref_line)
        scale = max(abs(crossed), abs(reference), 1e-99)
        self.assertLessEqual(
            abs(crossed - reference) / scale, self.tolerance,
            '%s crossed (D=%s) disagrees with %s: crossed=%r reference=%r'
            % (base_line, perm, ref_line, crossed, reference))

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
        # (row D over the leaves, reference line, crossed leaf signature); the
        # base leaves are [g,u,b,w+,t~,u]: particle 1, then particle 2, swapped
        # with the final u, the decay leaves b w+ never moving.
        # The last is a 3-cycle (the u stays incoming in slot 1, the final u
        # comes in as the u~ of slot 2, the g goes out in slot 6).
        cases = [
            ((5, 1, 2, 3, 4, 0), 'u~ u > t t~ g, t > b w+',
             (-2, 2, 5, 24, -6, 21)),
            ((0, 5, 2, 3, 4, 1), 'g u~ > t t~ u~, t > b w+',
             (21, -2, 5, 24, -6, -2)),
            ((1, 5, 2, 3, 4, 0), 'u u~ > t t~ g, t > b w+',
             (2, -2, 5, 24, -6, 21)),
        ]
        for perm, ref_line, pdgs in cases:
            with self.subTest(perm=perm):
                self._assert_decay_crossing(base, base_line, ref_line, perm,
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
        # base leaves [u,u~,e+,e-,e+,e-,g]: particle 2 swapped with the g.
        cases = [
            ((0, 6, 2, 3, 4, 5, 1), 'u g > z z u, z > e+ e-',
             (2, 21, -11, 11, -11, 11, 2)),
        ]
        for perm, ref_line, pdgs in cases:
            with self.subTest(perm=perm):
                self._assert_decay_crossing(base, base_line, ref_line, perm,
                                            pdgs)

    def test_decay_chain_crossings_survive_matrix_element_combination(self):
        """Identical decay-chain matrix elements must not merge their records.

        Without flavor grouping, g c > z c, z > e+ e- has the matrix element of
        g u > z u, z > e+ e-, and the decay-chain combination used to fold it
        into the u one, keeping its processes only: the crossings recorded on
        it (c c~ > z g, g c~ > z c~, ...) were then in no crossing table. The
        c directory must be written, and serve them.
        """
        outdir = pjoin(self.tmpdir, 'Proc_dc_nogroup')
        self.cmd.exec_cmd('set automatic_html_opening False')
        self.cmd.exec_cmd('set group_subprocesses False')
        self.cmd.exec_cmd('set apply_flavor_grouping False')
        self.cmd.exec_cmd('import model sm')
        self.cmd.exec_cmd('generate p p > z j, z > e+ e- %s' % _pin_crossing(''))
        self.cmd.exec_cmd('output standalone_fortran %s -f' % outdir)
        subproc = pjoin(outdir, 'SubProcesses')
        pdirs = sorted(name for name in os.listdir(subproc)
                       if name.startswith('P'))
        self.assertEqual([name.split('_', 1)[1] for name in pdirs],
                         ['gc_zc_z_epem', 'gd_zd_z_epem', 'gs_zs_z_epem',
                          'gu_zu_z_epem'])
        base_line = 'g c > z c, z > e+ e-'
        base = pjoin(subproc, pdirs[0])
        self._write_driver(base)
        self._build(base)
        # base leaves [g,c,e+,e-,c]; the recorded crossings, not every
        # applicable row (no --crossing_table=all).
        cases = [
            ((1, 4, 2, 3, 0), 'c c~ > z g, z > e+ e-', (4, -4, -11, 11, 21)),
            ((0, 4, 2, 3, 1), 'g c~ > z c~, z > e+ e-',
             (21, -4, -11, 11, -4)),
        ]
        for perm, ref_line, pdgs in cases:
            with self.subTest(perm=perm):
                self._assert_decay_crossing(base, base_line, ref_line, perm,
                                            pdgs)

    # SMATRIX at FLAV_IDX F1 NTRAIN times (training the shared good-helicity
    # filter at the base point), then once at FLAV_IDX F2 at the crossed point;
    # stdin: F1 F2 NTRAIN, then the base and the crossed momenta (E px py pz).
    _TRAINING_DRIVER = '''      PROGRAM TRAINCHECK
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      REAL*8 PB(0:3,NEXTERNAL), PC(0:3,NEXTERNAL), ANS
      INTEGER I, K, F1, F2, NTRAIN
      CALL SETPARA('param_card.dat')
      READ(*,*) F1, F2, NTRAIN
      DO K=1,NEXTERNAL
        READ(*,*) (PB(I,K),I=0,3)
      ENDDO
      DO K=1,NEXTERNAL
        READ(*,*) (PC(I,K),I=0,3)
      ENDDO
      DO I=1,NTRAIN
        CALL SMATRIX(PB, F1, ANS)
      ENDDO
      CALL SMATRIX(PC, F2, ANS)
      WRITE(*,'(A,ES24.16)') ' ANS=', ANS
      END
'''

    def test_massive_leg_crossing_after_base_training(self):
        """A crossing whose base has a massive leg survives the base training
        the shared good-helicity filter first.

        `u b1 > c1 d1` (b1 = g u~, c1 = u z, d1 = z g) folds u u~ > z g onto
        u g > u z. Four base rows are exact zeros in the u g frame only -- the z
        is massive, its helicity states mix under a boost -- and their sign-flip
        images are rows the crossing needs (two carry the z's helicity 0). The
        filter is shared by every crossing of a flavor, so 40 base calls used
        to freeze it without them: the crossed matrix element then came out
        low (-2.2e-4 at this point), while asked first it was exact. Training
        now closes the filter over the helicity of massive legs (HMV).
        """
        outdirs = {}
        for name, options in (('fold', '--use_crossing=True'),
                              ('exp', '--use_crossing=False')):
            outdir = pjoin(self.tmpdir, 'uz_train_%s' % name)
            self.cmd.exec_cmd('set automatic_html_opening False')
            self.cmd.exec_cmd('set group_subprocesses False')
            self.cmd.exec_cmd('import model sm')
            self.cmd.exec_cmd('define b1 = g u~')
            self.cmd.exec_cmd('define c1 = u z')
            self.cmd.exec_cmd('define d1 = z g')
            self.cmd.exec_cmd('generate u b1 > c1 d1 QED=1 QCD=1 %s' % options)
            self.cmd.exec_cmd('output standalone_fortran %s -f' % outdir)
            outdirs[name] = pjoin(outdir, 'SubProcesses')

        def pdir(root, suffix):
            found = [pjoin(root, d) for d in sorted(os.listdir(root))
                     if d.startswith('P') and d.endswith(suffix)]
            self.assertEqual(len(found), 1, '%s: %s' % (suffix, found))
            return found[0]
        self.assertFalse(any(d.endswith('_uux_zg')
                             for d in os.listdir(outdirs['fold'])),
                         'u u~ > z g was not folded onto u g > u z')
        fold, exp = pdir(outdirs['fold'], '_ug_uz'), pdir(outdirs['exp'],
                                                          '_uux_zg')
        for path in (fold, exp):
            with open(pjoin(path, 'check_sa.f'), 'w') as fsock:
                fsock.write(self._TRAINING_DRIVER)
            self._build(path)

        mz, energy, theta = 91.188, 500.0, 0.25
        pz = (4 * energy ** 2 - mz ** 2) / (4 * energy)
        ez = math.sqrt(pz ** 2 + mz ** 2)

        def point(theta, z_slot):
            sin, cos = math.sin(theta), math.cos(theta)
            z = (ez, pz * sin, 0, pz * cos)
            other = (pz, -pz * sin, 0, -pz * cos)
            final = [z, other] if z_slot == 2 else [other, z]
            return [(energy, 0, 0, energy), (energy, 0, 0, -energy)] + final
        base_point = point(1.1, 3)     # u g > u z: the z is leg 4
        crossed_point = point(theta, 2)  # u u~ > z g: the z is leg 3

        def run(path, f1, f2, ntrain):
            lines = ['%d %d %d' % (f1, f2, ntrain)] + [
                ' '.join('%.17e' % x for x in mom)
                for mom in base_point + crossed_point]
            out = subprocess.run(['./check'], cwd=path, capture_output=True,
                                 input='\n'.join(lines) + '\n',
                                 text=True).stdout
            match = re.search(r'ANS=\s*(\S+)', out)
            self.assertTrue(match, 'no ANS from %s:\n%s' % (path, out))
            return float(match.group(1))
        reference = run(exp, 1, 1, 0)
        self.assertGreater(reference, 0.0)
        # FLAV_IDX 2 = crossing row 1 of the single base flavor
        for ntrain in (0, 40):
            with self.subTest(base_calls_first=ntrain):
                self.assertAlmostEqual(
                    run(fold, 1, 2, ntrain) / reference, 1.0, delta=1e-10,
                    msg='the folded u u~ > z g disagrees with its own output '
                        'after %d base calls' % ntrain)

        # SMATRIXHEL at a crossed index takes the crossed process's OWN code:
        # code by code, the folded u u~ > z g equals its expanded output
        # (the base-slot code it used to take matched none of the 24).
        for path in (fold, exp):
            with open(pjoin(path, 'check_sa.f'), 'w') as fsock:
                fsock.write(self._HELCODE_DRIVER)
            os.remove(pjoin(path, 'check'))
            self._build(path)

        def per_code(path, flav_idx):
            lines = ['%d 24' % flav_idx] + [
                ' '.join('%.17e' % x for x in mom) for mom in crossed_point]
            out = subprocess.run(['./check'], cwd=path, capture_output=True,
                                 input='\n'.join(lines) + '\n',
                                 text=True).stdout
            return [float(v) for v in re.findall(r'HEL=\s*\d+\s+(\S+)', out)]
        folded, alone = per_code(fold, 2), per_code(exp, 1)
        self.assertEqual(len(alone), 24)
        self.assertEqual(sum(1 for v in alone if v), 12)
        for code, (a, b) in enumerate(zip(folded, alone), 1):
            self.assertAlmostEqual(a, b, delta=1e-10 * max(abs(b), 1e-300),
                                   msg='SMATRIXHEL code %d: folded %r, '
                                       'expanded %r' % (code, a, b))

    # SMATRIXHEL for every helicity code 1..NCODE at the momenta read from
    # stdin; stdin: FLAV_IDX NCODE, then the momenta (E px py pz).
    _HELCODE_DRIVER = '''      PROGRAM HELCODECHECK
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      REAL*8 P(0:3,NEXTERNAL), ANS
      INTEGER I, K, FLAV_IDX, NCODE
      CALL SETPARA('param_card.dat')
      READ(*,*) FLAV_IDX, NCODE
      DO K=1,NEXTERNAL
        READ(*,*) (P(I,K),I=0,3)
      ENDDO
      DO I=1,NCODE
        CALL SMATRIXHEL(P, I, FLAV_IDX, ANS)
        WRITE(*,'(A,I6,1X,ES24.16)') ' HEL=', I, ANS
      ENDDO
      END
'''


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
        self.cmd.exec_cmd('generate %s --use_crossing=True' % process)
        self.cmd.exec_cmd('output standalone_fortran %s -f' % outdir)

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

    def _run_check(self, proc_line, exporter='standalone_fortran'):
        import madgraph.various.process_checks as process_checks
        # The C++/mg7 backends need a working C++ compiler + build toolchain;
        # the fortran one needs f2py. Skip (do not fail) when unavailable.
        if exporter != 'standalone_fortran':
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

    def test_check_crossing_command_mg7(self):
        """standalone (madmatrix) backend: same genuine-crossing
        agreement, evaluated at a prescribed phase-space point injected into the
        SIMD momenta buffer."""
        results, process_checks = self._run_check(
            'xq xq > xq xq', exporter='standalone')
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
        """An unknown madmatrix --simd must raise a clear InvalidCmd.

        No build: constructing the mg7 backend validates the choice up front.
        """
        import madgraph
        import madgraph.various.process_checks as process_checks
        procdef = self.cmd.extract_process('g u > g u')
        with self.assertRaises(madgraph.InvalidCmd) as ctx:
            process_checks.check_crossing(
                procdef, param_card=None,
                options={'energy': 1000.0, 'proc_line': 'g u > g u',
                         'exporter': 'standalone', 'simd': 'not_a_simd'},
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
        cmd.exec_cmd(
            ('generate %s %s' % (process, _pin_crossing(options))).strip())
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
        for fmt in ('standalone_fortran', 'standalone'):
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

    def test_crossing_breaking_process_keeps_every_subprocess(self):
        """A process the exporter will not cross must not be folded either.

        A polarized leg breaks crossing symmetry (export_v4
        .breaks_crossing_symmetry), so the fortran standalone writes no
        crossing machinery for it. Its crossings used to be recorded at
        generation all the same: p p > z{0} j came out as the single g q
        directory, the q q~ and g q~ subprocesses reachable from nowhere. The
        same holds for a polarized leg inside a decay chain. Both must now come
        out exactly as an uncrossed generation does.
        """
        for proc in ('p p > z{0} j', 'p p > z j, z > e+{L} e-'):
            with self.subTest(process=proc):
                tag = 'pol_dc' if ',' in proc else 'pol'
                on = self._output('standalone_fortran', '%s_on' % tag,
                                  process=proc)
                off = self._output('standalone_fortran', '%s_off' % tag,
                                   process=proc,
                                   options='--use_crossing=False')
                subs_on = self._subprocesses(on)
                self.assertEqual(subs_on, self._subprocesses(off))
                # Guard the guard: both sides would agree if both had lost
                # the crossed subprocesses.
                self.assertGreater(len(subs_on), 1,
                                   'expected several subprocesses, got %s'
                                   % subs_on)

    def test_supported_outputs_accept_crossing(self):
        """Outputs that DO implement crossing must not be caught.

        Anchors the gate against being over-broad: a check that refused every
        output would pass both tests above. The fortran standalone decodes the
        extended FLAV_IDX directly; the grouped madevent output reaches the
        crossed subprocesses through the crossing router.
        """
        for fmt in ('standalone_fortran', 'madevent'):
            with self.subTest(format=fmt):
                self._output(fmt, 'ok_%s' % fmt)


class TestCrossingOutputOrder(unittest.TestCase):
    """An output must not depend on the outputs written before it.

    One generation with recorded crossings (merge_crossing='record') comes out
    folded into its bases for a folding standalone backend and expanded into
    explicit subprocesses for every other output. Both are built from the same
    self._curr_amps, so writing one output must leave that generation -- and
    the matrix elements the next output may reuse -- fit for the other kind.
    It did not: the expansion emptied the bases' crossed_processes in place,
    the ungrouped path replaced self._curr_amps by the expanded list, the
    amplitudes rebuilt from the matrix elements at the end of every output
    carry no crossing at all, and the cached matrix elements were reused
    whatever the crossing treatment they had been built for. A madevent output
    after a standalone_fortran one lost every crossed subprocess; a folding
    output after a madevent one folded nothing.

    pq pq > pq pq (pq = g u u~) is small and really folds: g g > g g,
    g g > u u~ and u u > u u carry every other subprocess. Every output is
    compared, file by file, with the same output written first in a fresh
    session. No 'import model' line: the define imports the Standard Model on
    its own, as it does for a script starting that way.

    Decay chains have a trap of their own: building the matrix elements pops
    the decay chains off the DecayChainAmplitude they come from, so an output
    built straight from self._curr_amps leaves the generation's chain without
    its decays. Once self._curr_amps was no longer replaced at the end of an
    output, the next output rebuilt from it wrote pq pq > z pq, z > e+ e- as
    the undecayed pq pq > z pq, and nothing complained. Covered on its own
    (DECAY_CHAIN) and next to a folding process, where the decay chain itself
    records nothing (MIXED).
    """

    SETUP = ('define pq = g u u~',)
    GENERATION = ('generate pq pq > pq pq --use_crossing=True',)
    DECAY_CHAIN = ('generate pq pq > z pq, z > e+ e- --use_crossing=True',)
    MIXED = GENERATION + ('add process u u~ > z g, z > e+ e-',)
    # Drops the cached matrix elements but keeps the generation, so the next
    # output is rebuilt from self._curr_amps. (set group_subprocesses cannot
    # be used for this: it drops the generation as well.)
    REBUILD = 'set loop_optimized_output False'
    UNGROUPED = ('set group_subprocesses False',)

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix='cross_order_')
        cls._fresh = {}
        cls._count = 0

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.tmpdir):
            shutil.rmtree(cls.tmpdir)

    def _session(self, steps, setup=(), generation=None):
        """One interface running the `generation` lines (GENERATION by
        default), then `steps` in turn: a 'set ...' line is executed, anything
        else is an output format written to a new directory. Returns the
        output directories, in order."""
        cmd = cmd_interface.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False')
        for line in tuple(setup) + self.SETUP + \
                tuple(generation or self.GENERATION):
            cmd.exec_cmd(line)
        outs = []
        for step in steps:
            if step.startswith('set '):
                cmd.exec_cmd(step)
                continue
            type(self)._count += 1
            out = pjoin(self.tmpdir, '%s_%d' % (step, self._count))
            cmd.exec_cmd('output %s %s -f -nojpeg' % (step, out))
            outs.append(out)
        return outs

    def _fresh_output(self, fmt, setup=(), generation=None):
        key = (fmt, tuple(setup), tuple(generation or self.GENERATION))
        if key not in self._fresh:
            self._fresh[key] = self._session([fmt], setup, generation)[0]
        return self._fresh[key]

    @staticmethod
    def _subprocesses(out_dir):
        """{P directory: {file: content}} over the regular files of every
        subprocess directory. FLV_<n> coupling names are numbered by a
        counter shared across the session, which depends on the outputs
        written before whatever the crossing does (--use_crossing=False drifts
        the same way on a decay chain), so the number is blanked."""
        path = pjoin(out_dir, 'SubProcesses')
        result = {}
        for pdir in sorted(os.listdir(path)):
            if not pdir.startswith('P'):
                continue
            files = {}
            for name in sorted(os.listdir(pjoin(path, pdir))):
                fpath = pjoin(path, pdir, name)
                if os.path.islink(fpath) or not os.path.isfile(fpath):
                    continue
                with open(fpath, 'rb') as stream:
                    text = stream.read().decode('utf-8', 'replace')
                files[name] = re.sub(r'\bFLV_\d+\b', 'FLV_n', text)
            result[pdir] = files
        return result

    def _assert_same_output(self, out_dir, fresh_dir, label):
        got = self._subprocesses(out_dir)
        want = self._subprocesses(fresh_dir)
        self.assertEqual(sorted(got), sorted(want),
                         '%s: not the subprocess directories of a fresh '
                         'output' % label)
        for pdir in want:
            self.assertEqual(sorted(got[pdir]), sorted(want[pdir]),
                             '%s: %s holds other files than in a fresh '
                             'output' % (label, pdir))
            for name in want[pdir]:
                self.assertTrue(got[pdir][name] == want[pdir][name],
                                '%s: %s/%s differs from a fresh output'
                                % (label, pdir, name))

    def _assert_order_independent(self, sequences, setup=(), generation=None):
        """Each sequence's last output must equal a fresh one of its format."""
        for steps in sequences:
            fmt = [s for s in steps if not s.startswith('set ')][-1]
            label = ' ; '.join(steps)
            with self.subTest(sequence=label):
                out = self._session(steps, setup, generation)[-1]
                self._assert_same_output(
                    out, self._fresh_output(fmt, setup, generation), label)

    def test_fresh_outputs_fold_and_expand(self):
        """Guard the guards: the process has to fold for the tests below to
        mean anything, and the two kinds of output must really differ."""
        folded = self._subprocesses(self._fresh_output('standalone_fortran'))
        expanded = self._subprocesses(self._fresh_output('madevent'))
        self.assertTrue(any('Crossed processes (folded into this matrix '
                            'element)' in files.get('check_sa.f', '')
                            for files in folded.values()),
                        'expected folded crossings in %s' % sorted(folded))
        self.assertLess(len(folded), len(expanded),
                        'expected the madevent output to expand the folded '
                        'crossings: %s vs %s' % (sorted(folded),
                                                 sorted(expanded)))

    def test_folding_output_after_grouped_output(self):
        """A folding output written after the grouped madevent one."""
        sequences = []
        for fmt in ('standalone_fortran', 'standalone'):
            sequences.append(('madevent', fmt))
            sequences.append(('madevent', self.REBUILD, fmt))
        self._assert_order_independent(sequences)

    def test_grouped_output_after_folding_output(self):
        """The grouped madevent output written after a folding one."""
        self._assert_order_independent(
            [('standalone_fortran', 'madevent'),
             ('standalone_fortran', self.REBUILD, 'madevent')])

    def test_ungrouped_output_order(self):
        """The same, with the ungrouped madevent output (the ungrouped path of
        the expansion)."""
        self._assert_order_independent(
            [('madevent', 'standalone_fortran'),
             ('madevent', self.REBUILD, 'standalone_fortran'),
             ('standalone_fortran', self.REBUILD, 'madevent')],
            setup=self.UNGROUPED)

    def test_decay_chain_outputs_keep_the_decay(self):
        """Guard: the decay chain folds, and keeps its decay, when written
        first."""
        folded = self._subprocesses(self._fresh_output(
            'standalone_fortran', generation=self.DECAY_CHAIN))
        self.assertTrue(folded and all('_z_' in pdir for pdir in folded),
                        'expected decayed subprocesses: %s' % sorted(folded))
        self.assertTrue(any('Crossed processes (folded into this matrix '
                            'element)' in files.get('check_sa.f', '')
                            for files in folded.values()),
                        'expected folded crossings in %s' % sorted(folded))

    def test_decay_chain_output_order(self):
        """A decay chain with recorded crossings, rebuilt by a later output of
        either kind."""
        self._assert_order_independent(
            [('standalone_fortran', self.REBUILD, 'standalone_fortran'),
             ('standalone_fortran', 'madevent', 'standalone_fortran'),
             ('standalone', self.REBUILD, 'standalone'),
             ('madevent', self.REBUILD, 'madevent')],
            generation=self.DECAY_CHAIN)

    def test_decay_chain_ungrouped_output_order(self):
        """The same on the ungrouped path."""
        self._assert_order_independent(
            [('standalone_fortran', self.REBUILD, 'standalone_fortran'),
             ('madevent', 'standalone_fortran', 'madevent')],
            setup=self.UNGROUPED, generation=self.DECAY_CHAIN)

    def test_decay_chain_next_to_folding_process(self):
        """A decay chain recording no crossing, generated next to a process
        that records some: the generation is kept across outputs all the
        same."""
        self._assert_order_independent(
            [('madevent', 'standalone_fortran'),
             ('standalone_fortran', 'madevent'),
             ('madevent', self.REBUILD, 'madevent')],
            generation=self.MIXED)


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


class TestStandaloneMg7CrossSymmetry(unittest.TestCase):
    """standalone (madmatrix) must reproduce the crossing.

    The crossing reproduction test for the data-parallel madmatrix
    backend. The extended flavor id encodes K = id / nflav, a row of the
    crossing table, and flav = id % nflav (0-based, NFLAV=1 here). The tests
    look the id of a crossed process up by its PDGs (_crossed_ids, through the
    compiled flavorPDG of check_sa's crossing demo), never by its row number.
    The key extra check versus the scalar C++ backend is that DIFFERENT events
    in the SAME SIMD page may carry DIFFERENT crossings while sharing the
    reduced flavor: the per-event momentum permutation must not be vectorized.

    The whole check needs to build and run real C++/SIMD code; skipped (not
    failed) if the compiler or the madmatrix build toolchain is unavailable.
    """

    # the crossed processes of the g g > u u~ base the tests ask for
    PDG_GQX_GQX = (21, -2, 21, -2)     # g u~ > g u~
    PDG_QQ_GG = (2, -2, 21, 21)        # u u~ > g g
    IDENTITY = 0
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
    def _output_madmatrix(self, process, name, options='',
                               out_options='', color_basis=None):
        """Write the standalone (madmatrix) output for `process`, return its P* dir.

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
        cmd.exec_cmd(('generate %s %s' % (process, _pin_crossing(options))).strip())
        cmd.exec_cmd(('output standalone %s -f %s'
                      % (outdir, out_options)).strip())

        subproc_root = pjoin(outdir, 'SubProcesses')
        pdirs = [pjoin(subproc_root, d) for d in sorted(os.listdir(subproc_root))
                 if d.startswith('P') and os.path.isdir(pjoin(subproc_root, d))]
        self.assertEqual(len(pdirs), 1,
                         'Expected a single subprocess directory for %s, got %s'
                         % (process, pdirs))
        return pdirs[0]

    def _cpp_source(self, pdir):
        """The process-specific generated source: the crossing data lives in
        ProcessTables.h, the per-event momentum gather and the per-lane
        external calls in EvaluateDiagrams.inc and the crossed flavorPDG in
        CPPProcess.cc. (The generic crossing code is backend-owned, in
        backend/<variant>/SigmaKin.cc, and is the same for every process.)"""
        text = []
        for name in ('CPPProcess.cc', 'ProcessTables.h', 'EvaluateDiagrams.inc'):
            with open(pjoin(pdir, name)) as fsock:
                text.append(fsock.read())
        return '\n'.join(text)

    def _patch_and_build(self, pdir):
        """Patch the shipped check_sa.cc so it can (a) also be asked for the
        first id PAST the crossing table (the shipped cap stops at the last
        row, see test_id_past_the_table_returns_zero) and (b) demonstrate a
        per-event mixed-crossing page (env MG_FLVMIX/MG_SAMEMOM), then build
        check_sa.exe. Skip if the madmatrix toolchain cannot build."""
        check = pjoin(pdir, 'check_sa.cc')
        with open(check) as fsock:
            src = fsock.read()
        cap = ('if( flavorID >= (unsigned int)( CPPProcess::nmaxflavor * '
               'CPPProcess::ncross ) )')
        self.assertEqual(src.count(cap), 1, 'the flavorID cap of check_sa.cc '
                         'is gone')
        src = src.replace(
            cap, 'if( flavorID > (unsigned int)( CPPProcess::nmaxflavor * '
                 'CPPProcess::ncross ) )')
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
        """Run check_sa.exe perf verbose and return the per-event ME list.

        A NaN is parsed as NaN rather than skipped: skipping it would shift
        every later event onto the wrong index of a mixed-crossing page."""
        run_env = dict(os.environ)
        if env:
            run_env.update(env)
        out = subprocess.check_output(
            ['./check_sa.exe', 'perf', '-v', '-f', str(flavor_id), '1', '8', '1'],
            cwd=pdir, env=run_env).decode()
        mes = [float(m) for m in
               re.findall(r'Matrix element =\s*(\S+)', out)]
        self.assertTrue(mes, 'no matrix element parsed from:\n%s' % out)
        return mes

    def _me(self, pdir, flavor_id):
        """First-event ME for a single (uniform) flavor id."""
        return self._event_mes(pdir, flavor_id)[0]

    def _crossed_ids(self, pdir):
        """{signed PDG tuple: extended flavor id} of the crossed processes
        check_sa's crossing demo shows (crossing_demo.dat), each with the PDGs
        the compiled flavorPDG reports for it. Needs check_sa.exe built."""
        out = subprocess.check_output(['./check_sa.exe'], cwd=pdir).decode()
        ids = {}
        for block in out.split(' flavorID ')[1:]:
            fid = int(block.split()[0])
            pdgs = tuple(int(line.split()[0]) for line in block.split('\n')
                         if re.match(r'\s+-?\d+\s+\S+e[+-]\d+', line))
            ids[pdgs] = fid
        self.assertTrue(ids, 'no crossed flavorID in the demo output of %s:\n%s'
                        % (pdir, out))
        return ids

    def _crossed_id(self, pdir, pdgs):
        ids = self._crossed_ids(pdir)
        self.assertIn(tuple(pdgs), ids, 'no crossed id evaluates %s in %s '
                      '(demo ids: %s)' % (pdgs, pdir, ids))
        return ids[tuple(pdgs)]

    @staticmethod
    def _table_size(pdir):
        """(ncross, nmaxflavor) the madmatrix module of `pdir` was written
        with."""
        with open(pjoin(pdir, 'ProcessTables.h')) as fsock:
            ncross = int(re.search(r'constexpr int ncross = (\d+);',
                                   fsock.read()).group(1))
        with open(pjoin(pdir, 'ProcessData.h')) as fsock:
            nflav = int(re.search(r'constexpr int nmaxflavor = (\d+);',
                                  fsock.read()).group(1))
        return ncross, nflav

    # Test-only knobs spliced into the output's copy of the backend SigmaKin.cc,
    # right after the good-helicity scan has built the per-crossing lists:
    #   MG_PADCROSS=c  drop the last good row of crossing c but keep the loop
    #                  bound, so the lanes of c reach a padding row (_hr = -1);
    #   MG_ONLYLAST=c  keep only that row, with the loop bound set to 1 (no
    #                  padding at all): its own contribution, on a uniform page.
    _PADDING_KNOBS = '''\
      if( const char* pc = getenv( "MG_PADCROSS" ) ) cNGoodPerCross[atoi( pc )] -= 1;
      if( const char* oc = getenv( "MG_ONLYLAST" ) )
      {
        const int c = atoi( oc );
        cGoodHelOfCross[c][0] = cGoodHelOfCross[c][cNGoodPerCross[c] - 1];
        cNGoodPerCross[c] = 1;
        cNGoodMaxCross = 1;
      }
'''

    def _add_padding_knobs(self, pdir):
        """Splice _PADDING_KNOBS into the cpu and simd SigmaKin.cc that the
        output of `pdir` compiles (whichever of the two the build picks)."""
        anchor = ('      for( int c = 0; c < cNcross; c++ ) if( cNGoodPerCross[c] '
                  '> cNGoodMaxCross ) cNGoodMaxCross = cNGoodPerCross[c];\n')
        backend = pjoin(os.path.dirname(os.path.dirname(pdir)), 'backend')
        for variant in ('cpu', 'simd'):
            path = pjoin(backend, variant, 'SigmaKin.cc')
            with open(path) as fsock:
                src = fsock.read()
            self.assertEqual(src.count(anchor), 1,
                             'the loop-bound line the knobs hook onto is gone '
                             'from %s' % path)
            src = '#include <cstdlib>\n' + src.replace(
                anchor, anchor + self._PADDING_KNOBS)
            with open(path, 'w') as fsock:
                fsock.write(src)

    def _output_pq_gg_qqx(self, name, options='', out_options=''):
        """Write the multiprocess `pq pq > pq pq` (pq = g u u~) and return its
        `g g > q q~` P* dir.

        `options` goes on the generate line (crossing pinned on unless it names
        --use_crossing itself), `out_options` on the output line. With the
        crossing on this is the FOLDED base of _output_folded_gg_qqx; with it
        off (on either line) it is the same process written on its own, next to
        the crossed subprocesses given back as directories of their own.

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
        cmd.exec_cmd('generate pq pq > pq pq %s' % _pin_crossing(options))
        cmd.exec_cmd(('output standalone %s -f %s'
                      % (outdir, out_options)).strip())

        subproc_root = pjoin(outdir, 'SubProcesses')
        pdirs = [pjoin(subproc_root, d) for d in sorted(os.listdir(subproc_root))
                 if d.startswith('P') and 'gg_QQx' in d
                 and os.path.isdir(pjoin(subproc_root, d))]
        self.assertEqual(len(pdirs), 1,
                         'expected exactly one g g > q q~ dir, got %s' % pdirs)
        return pdirs[0]

    def _output_folded_gg_qqx(self, name):
        """Write a multiprocess in which `g g > q q~` is the FOLDED base of its
        crossings, and return that P* dir.

        The crossing table only holds the crossings this ME actually
        records (every applicable one only with --crossing_table=all), so a
        crossed matrix element can only be asked for on a base that folded it
        in -- and a base that folds nothing is written without
        the crossing machinery at all (see
        test_nothing_folded_drops_the_machinery). A bare `generate u u~ > g g`
        records nothing, so the crossings below have to come from a real
        multiparticle expansion: `pq pq > pq pq` with pq = g u u~ folds
        `g u~ > g u~` and `u u~ > g g` onto the `g g > q q~` base -- the same
        two directions the standalone references below compute on their own.
        """
        pdir = self._output_pq_gg_qqx(name, options='--use_crossing=True')
        demo = pjoin(pdir, 'crossing_demo.dat')
        self.assertTrue(os.path.exists(demo),
                        'no crossing was folded onto %s' % pdir)
        with open(pjoin(pdir, 'ProcessTables.h')) as fsock:
            self.assertTrue('use_crossing = true' in fsock.read(),
                            'the folded base %s was written without the '
                            'crossing machinery' % pdir)
        return pdir

    # ------------------------------------------------------------------
    def test_gg_qqx_crossed_gives_qg_qg(self):
        """g g > q q~ crossed to g u~ > g u~ equals it at the same momenta
        (both 2->2 massless -> identical RAMBO momenta for the same seed).

        The base must be one that FOLDED this crossing in: its crossing table
        holds the recorded crossings, so a bare `generate u u~ > g g` (which
        records none) has no crossed row to ask for unless it is written with
        --crossing_table=all. See _output_folded_gg_qqx."""
        crossed = self._output_folded_gg_qqx('ggqqx')
        reference = self._output_madmatrix(PROC_GQX_GQX, 'gqxgqx',
                                                color_basis='trace')
        self._patch_and_build(crossed)
        self._patch_and_build(reference)

        crossed_val = self._me(crossed,
                               self._crossed_id(crossed, self.PDG_GQX_GQX))
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
        reference = self._output_madmatrix(PROC_QQ_GG, 'qqgg_rev',
                                                color_basis='trace')
        self._patch_and_build(crossed)
        self._patch_and_build(reference)
        # the u u~ > g g row of the g g > u u~ base is not an involution (the
        # diagram pairing of the recorded crossing makes it a 4-cycle), so this
        # is the madmatrix case where reading a row in the wrong view shows
        import madgraph.various.process_checks as process_checks
        crossed_id = self._crossed_id(crossed, self.PDG_QQ_GG)
        row = process_checks._mg7_crossing_rows(crossed)[
            crossed_id // self._table_size(crossed)[1]]
        self.assertTrue(any(row[row[k]] != k for k in range(len(row))),
                        'the u u~ > g g row %s is an involution: the fixture '
                        'no longer tells the two views apart' % (row,))
        reference_val = self._me(reference, self.IDENTITY)
        self.assertAlmostEqual(
            self._me(crossed, crossed_id),
            reference_val,
            delta=self.tolerance * abs(reference_val),
            msg='g g > q q~ crossed != u u~ > g g identity')

    def test_per_event_different_cross(self):
        """THE point of the SIMD port: within ONE SIMD page, events carrying
        DIFFERENT crossings (but the same reduced flavor) each get their own
        crossed matrix element. Feed identical momenta to every event, alternate
        the crossing per event (even -> identity, odd -> the g u~ > g u~ row)
        and check
        each lane independently.

        The crossing used here is a RECORDED one of the folded base, which is
        what the crossing table holds (see _output_folded_gg_qqx)."""
        pdir = self._output_folded_gg_qqx('ggqqx_perevent')
        self._patch_and_build(pdir)
        crossed_id = self._crossed_id(pdir, self.PDG_GQX_GQX)
        identity_val = self._me(pdir, self.IDENTITY)
        crossed_val = self._me(pdir, crossed_id)
        self.assertNotAlmostEqual(identity_val, crossed_val, places=6,
                                  msg='degenerate: identity == crossed')
        mixed = self._event_mes(
            pdir, self.IDENTITY,
            env={'MG_SAMEMOM': '1',
                 'MG_FLVMIX': '%d,%d' % (self.IDENTITY, crossed_id)})
        self.assertGreaterEqual(len(mixed), 4,
                                'need several events to prove per-event crossing')
        for i, me in enumerate(mixed):
            expected = identity_val if i % 2 == 0 else crossed_val
            self.assertAlmostEqual(
                me, expected, delta=self.tolerance * abs(expected) + 1e-12,
                msg='event %d (cross %s) got %r, expected %r'
                % (i, 'id' if i % 2 == 0 else 'g u~ > g u~', me, expected))

    def test_padding_helicity_row_contributes_zero(self):
        """A lane whose crossing has FEWER good helicities than the per-lane
        loop bound (cNGoodMaxCross) runs into padding rows (_hr = -1, every
        helicity mask 0), and must add exactly nothing there.

        The external block used to mask the momentum along with the
        wavefunction, so a padding lane had p = 0, its massless propagators
        evaluated 0/0 and its |M|^2 came out NaN. Nothing keeps the
        per-crossing counts equal (a row at ~1e-30 in one crossing can be an
        exact zero in another), so the padding row is forced: the list of
        the u u~ > g g row loses its last row after the scan while the loop
        bound stays.
        Those lanes must return the full value minus that row's contribution,
        measured on its own without any padding (MG_ONLYLAST), and the
        identity lanes sharing the page must not move."""
        pdir = self._output_folded_gg_qqx('ggqqx_padding')
        self._add_padding_knobs(pdir)
        self._patch_and_build(pdir)
        cross = self._crossed_id(pdir, self.PDG_QQ_GG)
        # the knobs take the crossing-table row: id / nmaxflavor
        row = cross // self._table_size(pdir)[1]
        identity_val = self._me(pdir, self.IDENTITY)
        full_val = self._me(pdir, cross)
        dropped_val = self._event_mes(pdir, cross,
                                      env={'MG_ONLYLAST': str(row)})[0]
        # Non-vacuous: the dropped row carries a visible share of |M|^2, so a
        # padding lane that silently kept it would fail too.
        self.assertGreater(dropped_val, 1e-3 * full_val,
                           'the dropped row (%r of %r) is too small to tell '
                           'the padded value apart' % (dropped_val, full_val))
        expected = full_val - dropped_val
        mixed = self._event_mes(
            pdir, self.IDENTITY,
            env={'MG_PADCROSS': str(row), 'MG_SAMEMOM': '1',
                 'MG_FLVMIX': '%d,%d' % (self.IDENTITY, cross)})
        self.assertGreaterEqual(len(mixed), 4,
                                'need several events to mix crossings in a page')
        for i, me in enumerate(mixed):
            want = identity_val if i % 2 == 0 else expected
            self.assertFalse(math.isnan(me),
                             'event %d (cross %s) is NaN: the padding row was '
                             'not a clean zero' % (i, 'id' if i % 2 == 0 else cross))
            self.assertAlmostEqual(
                me, want, delta=self.tolerance * abs(want) + 1e-12,
                msg='event %d (cross %s) got %r, expected %r'
                % (i, 'id' if i % 2 == 0 else cross, me, want))

    def test_id_past_the_table_returns_zero(self):
        """An extended id past the last row of the crossing table names no
        crossing: its matrix element must come back as an exact 0 -- not an
        out-of-bounds read of the per-row good-helicity and C-parity tables,
        not an abort.

        Asked of a base that really folds crossings, since only such a base is
        written with the crossing machinery (one that folds nothing decodes no
        crossing at all). The shipped check_sa caps the id at the table, so
        _patch_and_build lets exactly one more row through."""
        import madgraph.various.process_checks as process_checks
        pdir = self._output_folded_gg_qqx('ggqqx_inv')
        ncross, nflav = self._table_size(pdir)
        self.assertEqual(process_checks._mg7_compiled_crossings(pdir),
                         set(range(ncross)),
                         'check crossing would not ask %s for every row of '
                         'its table' % pdir)
        self._patch_and_build(pdir)
        # Guard the guard: a build that returned 0 for everything would pass.
        self.assertNotEqual(self._me(pdir, self.IDENTITY), 0.0,
                            'degenerate: the identity ME is already 0')
        self.assertEqual(self._me(pdir, ncross * nflav), 0.0,
                         'an id past the crossing table must give a zero ME')

    def test_use_crossing_false_byte_identical(self):
        """--use_crossing=False must emit NO crossing machinery (every crossing
        token absent from the generated source) and still give the same
        uncrossed matrix element as the crossing-on build. (A full byte-identical
        `diff -r` against the pre-feature output was checked by hand; here we
        assert the token absence and the numerical invariance.)

        The crossing-on side is a base that really folds crossings: one that
        folds nothing is written without the machinery whatever the flag says
        (test_nothing_folded_drops_the_machinery)."""
        on_dir = self._output_folded_gg_qqx('ggqqx_on')
        off_dir = self._output_pq_gg_qqx('ggqqx_off',
                                         options='--use_crossing=False')
        on_src = self._cpp_source(on_dir)
        off_src = self._cpp_source(off_dir)
        self.assertIn('use_crossing = true', on_src)
        self.assertIn('use_crossing = false', off_src)
        for token in ('cross_gather( xcr', 'base_pdg',
                      'cGoodHelOfCross', 'xmom', 'icsign'):
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

        On a base that really folds crossings: one that folds nothing is
        written without the machinery whatever either flag says, which would
        make the equality below hold for the wrong reason.
        """
        gen_dir = self._output_pq_gg_qqx('ggqqx_genoff',
                                         options='--use_crossing=False')
        out_dir = self._output_pq_gg_qqx('ggqqx_outoff',
                                         out_options='--use_crossing=False')
        out_src = self._cpp_source(out_dir)
        self.assertEqual(self._cpp_source(gen_dir), out_src,
                         '--use_crossing=False writes a different source on the '
                         'output line than on the generate line')
        # Guard the guard: an exporter that never emits the machinery would
        # satisfy the equality above with both sides broken.
        on_src = self._cpp_source(self._output_folded_gg_qqx('ggqqx_defaulton'))
        self.assertIn('use_crossing = true', on_src)
        self.assertIn('use_crossing = false', out_src)
        for token in ('cross_gather( xcr', 'base_pdg',
                      'cGoodHelOfCross', 'xmom'):
            self.assertIn(token, on_src,
                          '%s should be emitted with crossing on' % token)
            self.assertNotIn(token, out_src,
                             '%s must NOT survive --use_crossing=False on the '
                             'output line' % token)

    def test_gpu_backend_crosses_flavor_ids(self):
        """The GPU backend evaluates an extended flavor id as its crossing,
        as the cpu/simd ones do.

        It used to refuse them (umami_matrix_element counted the ids past
        nmaxflavor and returned UMAMI_ERROR_UNSUPPORTED_INPUT): a folded output
        could not run on a GPU, nor a gridpack made from one on cpu. No
        CUDA/HIP toolchain runs in this suite, so this checks the SHIPPED gpu
        source; the GPU CI (crossing_folding) checks the behaviour. Each GPU
        external call must read its momentum from the input slot of the
        thread's crossing row (xperm) with the base NSF sign of the cpu block
        above it times xic; calculate_jamps evaluates the thread's own
        crossing's good helicity, and the denominator and the reported
        helicity are the crossed ones."""
        # a base that really folds crossings: one folding nothing is written
        # on the plain path (use_crossing = false)
        pdir = self._output_folded_gg_qqx('ggqqx_gpu')
        gpu = pjoin(pdir, os.pardir, os.pardir, 'backend', 'gpu')
        with open(pjoin(gpu, 'umami.cc')) as fsock:
            umami = fsock.read()
        with open(pjoin(gpu, 'SigmaKin.cc')) as fsock:
            sigmakin = fsock.read()
        with open(pjoin(pdir, 'check_sa.cc')) as fsock:
            check_sa = fsock.read()
        with open(pjoin(pdir, 'EvaluateDiagrams.inc')) as fsock:
            diagrams = fsock.read()

        # assertTrue rather than assertIn/assertRegex: those print the whole file
        self.assertTrue(re.search(
            r'#ifdef MGONGPUCPP_GPUIMPL\n[^#]*int xperm\[npar\], xic\[npar\];\s*'
            r'cross_gather\( \(int\)\( iflavor_ext / \(unsigned int\)nmaxflavor \), xperm, xic \);',
            diagrams), 'EvaluateDiagrams.inc reads no crossing row on the GPU')
        # every crossed external block: cpu blend first, then the GPU call
        blocks = re.findall(
            r'(\w+xxxx)<M_ACCESS, W_ACCESS>\( xmom, [^;]*?, ([+-]\d), cFlavors\[iflavor\]\[(\d+)\], aloha_x\[0\]'
            r'.*?#else\n\s*(\w+xxxx)<M_ACCESS, W_ACCESS>\( momenta, ([^;]*)\);\n#endif',
            diagrams, re.S)
        self.assertTrue(blocks, 'no crossed external call found')
        for routine, sign, slot, gpu_routine, gpu_args in blocks:
            self.assertEqual(gpu_routine, routine)
            self.assertTrue(gpu_args.strip().endswith('xperm[%s]' % slot) and
                            ('%s * xic[%s]' % (sign, slot)) in gpu_args,
                            'GPU %s of slot %s is not crossed: %s'
                            % (routine, slot, gpu_args))
        self.assertTrue('ihel = crossed_hel_row( ihel, iflavor_ext );' in sigmakin,
                        'calculate_jamps does not evaluate the crossing\'s own helicity')
        self.assertTrue('iflavor_ext % (unsigned int)nmaxflavor' in sigmakin,
                        'the flavor tables are not indexed by the base flavor')
        self.assertTrue('/ ( (fptype)spincol_cross( dcr ) * (fptype)ident_cross( dcr, dfl ) )'
                        in sigmakin, 'normalise_output keeps the base denominator')
        self.assertTrue('allselhel[ievt] = selected_hel_code( row, iflavorVec[ievt] );'
                        in sigmakin, 'the reported helicity is not the crossed code')
        self.assertTrue('const bool validFlavor = use_crossing || fid < (unsigned int)nmaxflavor;'
                        in sigmakin, 'a crossing build gives a crossed id a NaN')
        self.assertFalse('n_bad_flavors' in umami or 'UMAMI_ERROR_UNSUPPORTED_INPUT;' in
                         umami.split('copy_inputs<<<')[1].split('sigmaKin(')[0],
                         'umami_matrix_element still refuses crossed flavor ids')
        self.assertFalse('not supported by the GPU backend' in check_sa,
                         'check_sa skips the crossing demo on a GPU build')

    def _assert_plain_path(self, pdir, label):
        """`pdir` was written without the crossing machinery."""
        # assertTrue, not assertIn: the latter would print the whole file.
        with open(pjoin(pdir, 'ProcessTables.h')) as fsock:
            self.assertTrue('use_crossing = false' in fsock.read(),
                            '%s: ProcessTables::use_crossing is not false'
                            % label)
        with open(pjoin(pdir, 'EvaluateDiagrams.inc')) as fsock:
            diagrams = fsock.read()
        for token in ('xmom', 'cGoodHelOfCross'):
            self.assertFalse(token in diagrams,
                             '%s: %s emitted into EvaluateDiagrams.inc'
                             % (label, token))
        self.assertFalse(os.path.exists(pjoin(pdir, 'crossing_demo.dat')),
                         '%s: a crossing demo was written' % label)
        # ... so `check crossing` must not ask it for anything but the identity
        import madgraph.various.process_checks as process_checks
        self.assertEqual(process_checks._mg7_compiled_crossings(pdir), set([0]),
                         '%s: check crossing would ask a plain build for a '
                         'crossing code' % label)

    def test_nothing_folded_drops_the_machinery(self):
        """--use_crossing=True on a process that folds nothing writes the
        plain path.

        A bare `u u~ > g g` records no crossed subprocess, so its crossing
        table would hold the identity alone: the per-lane
        crossing path (per-state external blend, per-event momentum gather,
        cNGoodMaxCross loop) could only ever recompute what the plain path
        computes. It used to be written all the same, being gated on the flag
        alone. Gated on the recorded crossings too, the output is exactly the
        --use_crossing=False one.
        """
        on_dir = self._output_madmatrix(PROC_QQ_GG, 'qqgg_nofold_on',
                                        options='--use_crossing=True')
        off_dir = self._output_madmatrix(PROC_QQ_GG, 'qqgg_nofold_off',
                                         options='--use_crossing=False')
        self._assert_plain_path(on_dir, PROC_QQ_GG)
        self.assertEqual(self._cpp_source(on_dir), self._cpp_source(off_dir),
                         'a crossing-on output folding nothing differs from '
                         'the --use_crossing=False one')

    def test_mg7_output_folds_the_crossings(self):
        """`output mg7` folds the crossings recorded at generation.

        mg7 is a folding format: each crossed subprocess gets a subprocesses.json
        entry of its own, evaluated by its base's library at the extended
        flavor id (TestMg7FoldedCrossing has the physics). The base directory is
        written with the machinery, and nothing else is written for the
        crossings -- not the plain path the flag-only gate once got wrong, and
        no directory of their own either.
        """
        outdir = pjoin(self.tmpdir, 'mg7_on')
        cmd = cmd_interface.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False')
        cmd.exec_cmd('set group_subprocesses False')
        cmd.exec_cmd('set apply_flavor_grouping True')
        cmd.exec_cmd('import model sm')
        cmd.exec_cmd('define xq = u u~')
        cmd.exec_cmd('generate xq xq > xq xq --use_crossing=True')
        # Guard the guard: the generation must really have folded crossings,
        # or there would be nothing for the output to fold.
        self.assertTrue(any(amp.get('crossed_processes')
                            for amp in cmd._curr_amps),
                        'xq xq > xq xq folded no crossing at generation')
        cmd.exec_cmd('output mg7 %s -f' % outdir)

        subproc_root = pjoin(outdir, 'SubProcesses')
        pdirs = [d for d in sorted(os.listdir(subproc_root))
                 if d.startswith('P') and os.path.isdir(pjoin(subproc_root, d))]
        with open(pjoin(subproc_root, 'subprocesses.json')) as fsock:
            entries = json.load(fsock)
        crossed = [e for e in entries if e.get('crossing')]
        self.assertTrue(crossed, 'no crossed subprocess entry was written')
        bases = set(e['crossing']['base'] for e in crossed)
        for base in bases:
            self.assertIn(os.path.basename(base), pdirs)
        self.assertEqual(len(pdirs), len(entries) - len(crossed),
                         'a crossed subprocess got a directory of its own: %s'
                         % pdirs)
        for base in bases:
            with open(pjoin(outdir, base, 'ProcessTables.h')) as fsock:
                self.assertTrue('use_crossing = true' in fsock.read(),
                                '%s folds crossings without the machinery'
                                % base)

    # Prints the selected helicity code (0-based, as umami reports it) of nevt
    # events at ONE phase-space point, the helicity random number on a uniform
    # grid: argv = flavor id, nevt, then npar*(E px py pz).
    _HEL_DRIVER = r"""
#include "umami.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main( int argc, char** argv )
{
  const unsigned int fid = atoi( argv[1] );
  const int nevt = atoi( argv[2] );
  UmamiHandle h = nullptr;
  if( umami_initialize( &h, "../../Cards/param_card.dat" ) != UMAMI_SUCCESS ) return 2;
  int npar = 0;
  umami_get_meta( UMAMI_META_PARTICLE_COUNT, &npar );
  std::vector<double> mom( (size_t)4 * npar * nevt ), rnd( nevt ), me( nevt ), as( nevt, 0.118 );
  std::vector<unsigned int> flv( nevt, fid );
  std::vector<int> hel( nevt, -1 );
  for( int ip = 0; ip < npar; ip++ )
    for( int i4 = 0; i4 < 4; i4++ )
      for( int ie = 0; ie < nevt; ie++ )
        mom[(size_t)i4 * npar * nevt + (size_t)ip * nevt + ie] = atof( argv[3 + 4 * ip + i4] );
  for( int ie = 0; ie < nevt; ie++ ) rnd[ie] = ( ie + 0.5 ) / nevt;
  UmamiInputKey ik[4] = { UMAMI_IN_MOMENTA, UMAMI_IN_FLAVOR_INDEX, UMAMI_IN_ALPHA_S, UMAMI_IN_RANDOM_HELICITY };
  const void* in[4] = { mom.data(), flv.data(), as.data(), rnd.data() };
  UmamiOutputKey ok[2] = { UMAMI_OUT_MATRIX_ELEMENT, UMAMI_OUT_HELICITY_INDEX };
  void* out[2] = { me.data(), hel.data() };
  if( umami_matrix_element( h, nevt, nevt, 0, 4, ik, in, 2, ok, out ) != UMAMI_SUCCESS ) return 3;
  for( int ie = 0; ie < nevt; ie++ ) printf( "%d\n", hel[ie] );
  umami_free( h );
  return 0;
}
"""

    def _selected_helicities(self, pdir, flavor_id, momenta, nevt):
        """Build _HEL_DRIVER as the check_sa.exe of `pdir` and return the codes
        it reports for `nevt` events at `momenta`."""
        check = pjoin(pdir, 'check_sa.cc')
        os.remove(check)   # a link to the shared SubProcesses/check_sa.cc
        with open(check, 'w') as fsock:
            fsock.write(self._HEL_DRIVER)
        with open(os.devnull, 'w') as devnull:
            rc = subprocess.call(['make', '-j2', 'check_sa.exe'], cwd=pdir,
                                 stdout=devnull, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, FPTYPE='d'))
        if rc != 0:
            self.skipTest('madmatrix build toolchain unavailable (make failed)')
        out = subprocess.check_output(
            ['./check_sa.exe', str(flavor_id), str(nevt)]
            + ['%.17g' % x for p in momenta for x in p], cwd=pdir).decode()
        return [int(code) for code in out.split()]

    def test_moved_leg_reports_its_own_helicity(self):
        """A crossed event reports the crossed process's OWN helicity code --
        the code its expanded output reports, and the one the madevent output
        writes -- also when the crossing moves a leg into a slot with other
        helicity states.

        `u b1 > c1 d1` (b1 = g u~, c1 = u z, d1 = z g) folds u u~ > z g onto
        u g > u z: the z lands in the base's u slot, the g in its z slot. The
        code once ran over the BASE slot's states, which have no digit for the
        z's helicity 0 (its longitudinal events came out transverse), and
        until the madevent convention was adopted it was a base-relative code
        that only a per-crossing table could decode. At one phase-space point
        and the same random numbers, the folded crossing now reports every
        code exactly as often as u u~ > z g written on its own, and the codes
        decode with the crossed process's own helicity table.
        """
        outdirs = {}
        for name, options in (('fold', ''), ('exp', ' --use_crossing=False')):
            outdir = pjoin(self.tmpdir, 'uz_%s' % name)
            cmd = cmd_interface.MasterCmd()
            cmd.no_notification()
            cmd.exec_cmd('set automatic_html_opening False')
            cmd.exec_cmd('set group_subprocesses False')
            cmd.exec_cmd('import model sm')
            cmd.exec_cmd('define b1 = g u~')
            cmd.exec_cmd('define c1 = u z')
            cmd.exec_cmd('define d1 = z g')
            cmd.exec_cmd('generate u b1 > c1 d1 QED=1 QCD=1 --use_crossing=True')
            cmd.exec_cmd('output standalone %s -f%s' % (outdir, options))
            outdirs[name] = pjoin(outdir, 'SubProcesses')

        def pdirs(root, suffix):
            return [pjoin(root, d) for d in sorted(os.listdir(root))
                    if d.startswith('P') and d.endswith(suffix)]
        self.assertEqual(pdirs(outdirs['fold'], '_uux_zg'), [],
                         'u u~ > z g was not folded onto u g > u z')
        (fold,), (exp,) = pdirs(outdirs['fold'], '_ug_uz'), \
            pdirs(outdirs['exp'], '_uux_zg')
        with open(pjoin(fold, 'crossing_demo.dat')) as fsock:
            ids = [int(i) for i in fsock.read().split()]
        self.assertEqual(len(ids), 1, ids)
        with open(pjoin(exp, 'ProcessData.h')) as fsock:
            thel = fsock.read().split('tHel')[1].split(';')[0]
        rows = [tuple(int(v) for v in row) for row in
                re.findall(r'\{\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\s*\}',
                           thel)]

        # u u~ > z g at sqrt(s) = 1 TeV, theta = 0.7
        mz, energy, theta = 91.188, 500.0, 0.7
        pz = (4 * energy ** 2 - mz ** 2) / (4 * energy)
        ez = math.sqrt(pz ** 2 + mz ** 2)
        sin, cos = math.sin(theta), math.cos(theta)
        momenta = [(energy, 0, 0, energy), (energy, 0, 0, -energy),
                   (ez, pz * sin, 0, pz * cos), (pz, -pz * sin, 0, -pz * cos)]
        nevt = 20000
        folded = self._selected_helicities(fold, ids[0], momenta, nevt)
        alone = self._selected_helicities(exp, 0, momenta, nevt)
        longitudinal = sum(1 for code in alone if rows[code][2] == 0)
        self.assertTrue(longitudinal > 0, 'no longitudinal z at this point')
        self.assertEqual(sum(1 for code in folded if rows[code][2] == 0),
                         longitudinal, 'the folded crossing mis-reports the '
                         'helicity of the z it moved into the u slot')
        for code in set(folded) | set(alone):
            self.assertLessEqual(
                abs(folded.count(code) - alone.count(code)), 2,
                'code %d %s: reported %d times folded, %d times on its own'
                % (code, rows[code], folded.count(code), alone.count(code)))


class TestMg7FoldedCrossing(unittest.TestCase):
    """`output mg7` folds the crossings recorded at generation.

    Each crossed subprocess becomes a subprocesses.json entry of its own,
    evaluated by its base's library at the extended flavor id K*nflav + flav
    (export_mg7.OneProcessExporterMG7.get_crossed_subprocess_info): what it
    integrates must be exactly what the expanded output (--use_crossing=False
    on the output line, same generation) integrates, down to the slot order of
    every flavor row and the channels. The structure checks need no runtime;
    the cross-section ones need madspace and LHAPDF and skip without them.
    """

    debugging = getattr(unittest, 'debug', False)

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='cross_mg7fold_')

    def tearDown(self):
        if not self.debugging and os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _outputs(self, process, name, defines=(), setup=()):
        """Generate `process` with the crossing recorded and write it twice:
        folded (`output mg7`) and expanded (`output mg7 --use_crossing=False`).
        Returns the two output directories."""
        cmd = cmd_interface.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False')
        cmd.exec_cmd('set apply_flavor_grouping True')
        for line in setup:
            cmd.exec_cmd(line)
        cmd.exec_cmd('import model sm')
        for define in defines:
            cmd.exec_cmd('define %s' % define)
        cmd.exec_cmd('generate %s --use_crossing=True' % process)
        self.assertTrue(cmd._has_recorded_crossings(cmd._curr_amps),
                        '%s folded no crossing at generation' % process)
        folded = pjoin(self.tmpdir, name + '_folded')
        expanded = pjoin(self.tmpdir, name + '_expanded')
        cmd.exec_cmd('output mg7 %s -f' % folded)
        cmd.exec_cmd('output mg7 %s -f --use_crossing=False' % expanded)
        return folded, expanded

    @staticmethod
    def _entries(outdir):
        with open(pjoin(outdir, 'SubProcesses', 'subprocesses.json')) as fsock:
            return json.load(fsock)

    @staticmethod
    def _pdirs(outdir):
        root = pjoin(outdir, 'SubProcesses')
        return [d for d in sorted(os.listdir(root))
                if d.startswith('P') and os.path.isdir(pjoin(root, d))]

    @staticmethod
    def _rows(entries):
        """Every (slot-ordered flavor row, mirror) the entries integrate."""
        return sorted((tuple(option), bool(flavor['mirror']))
                      for entry in entries for flavor in entry['flavors']
                      for option in flavor['options'])

    @staticmethod
    def _nflav(outdir, base):
        with open(pjoin(outdir, base, 'ProcessData.h')) as fsock:
            match = re.search(r'nmaxflavor\s*=\s*(\d+)', fsock.read())
        return int(match.group(1))

    @staticmethod
    def _topologies(entries):
        """The channel diagrams of `entries` as the phase space sees them:
        (topology, permutation, propagator pdgs), diagram numbers aside (a
        crossed entry numbers them as its base). Per diagram, not per channel:
        the entry of one crossing-table row keeps only the diagrams its
        flavors have."""
        return sorted(set(
            (json.dumps(c['propagators']), json.dumps(c['vertices']),
             json.dumps(c['on_shell_propagators']),
             json.dumps(d['permutation']), json.dumps(d['propagator_pdgs']))
            for e in entries for c in e['channels'] for d in c['diagrams']))

    @staticmethod
    def _flows(entries):
        """The colour flows of `entries`, their labels renumbered in order of
        appearance (a crossed flow keeps its base's labels)."""
        def canonical(flow):
            labels = {}
            return json.dumps([[labels.setdefault(c, 501 + len(labels))
                                if c else 0 for c in leg] for leg in flow])
        return sorted(set(canonical(flow) for e in entries
                          for flow in e['color_flows']))

    def _check_folding(self, folded, expanded):
        """The folded output integrates what the expanded one does: per
        crossed process, the same flavor rows (slot order and mirror
        included), helicities, colour flows and channels as the expanded
        output's own entries for it, evaluated by its base's library; and the
        expanded directories carry no crossing machinery."""
        fentries, eentries = self._entries(folded), self._entries(expanded)
        crossed = [e for e in fentries if e.get('crossing')]
        self.assertTrue(crossed, 'no crossed subprocess entry was written')
        self.assertEqual(self._rows(fentries), self._rows(eentries),
                         'the folded output does not integrate the flavor '
                         'rows of the expanded one')
        self.assertEqual(len(self._pdirs(folded)),
                         len(fentries) - len(crossed),
                         'a crossed subprocess got a directory of its own')
        for pdir in self._pdirs(expanded):
            path = pjoin(expanded, 'SubProcesses', pdir)
            with open(pjoin(path, 'ProcessTables.h')) as fsock:
                self.assertTrue('use_crossing = false' in fsock.read(),
                                'expanded %s has the crossing machinery' % pdir)
            self.assertFalse(os.path.exists(pjoin(path, 'crossing_demo.dat')))

        def process(entry):
            return (tuple(entry['incoming']), tuple(entry['outgoing']))
        for key in sorted(set(process(e) for e in crossed)):
            mine = [e for e in crossed if process(e) == key]
            theirs = [e for e in eentries if process(e) == key]
            self.assertTrue(theirs, 'the expanded output has no %s' % (key,))
            self.assertEqual(self._rows(mine), self._rows(theirs), key)
            self.assertEqual(
                sorted(set(tuple(h) for e in mine for h in e['helicities'])),
                sorted(set(tuple(h) for e in theirs for h in e['helicities'])),
                '%s: not the helicities of the expanded output' % (key,))
            self.assertEqual(self._flows(mine), self._flows(theirs),
                             '%s: not the colour flows of the expanded output'
                             % (key,))
            self.assertEqual(self._topologies(mine), self._topologies(theirs),
                             '%s: not the channels of the expanded output'
                             % (key,))

        bases = dict((e['path'], e) for e in fentries if not e.get('crossing'))
        for entry in crossed:
            base = bases[entry['crossing']['base']]
            self.assertEqual(entry['me_path'], base['me_path'])
            self.assertEqual(entry['diagram_count'], base['diagram_count'])
            nflav = self._nflav(folded, base['path'])
            row = entry['crossing']['row']
            self.assertGreater(row, 0)
            for flavor in entry['flavors']:
                self.assertEqual(flavor['index'] // nflav, row,
                                 'flavor index %d is not on crossing row %d'
                                 % (flavor['index'], row))
            self.assertTrue(entry['channels'])
            for channel in entry['channels']:
                for diag in channel['diagrams']:
                    self.assertTrue(0 <= diag['diagram'] < base['diagram_count'])
                    self.assertTrue(diag['active_flavors'])
                    self.assertTrue(all(0 <= f < len(entry['flavors'])
                                        for f in diag['active_flavors']))
            self.assertEqual(len(entry['color_flows']),
                             len(base['color_flows']),
                             'the colour flows are not indexed by the base flow')
            with open(pjoin(folded, base['path'], 'ProcessTables.h')) as fsock:
                self.assertTrue('use_crossing = true' in fsock.read())
        return fentries, eentries, crossed

    def test_w_jet_entries_match_the_expanded_output(self):
        """p p > w+ j: one directory folding two crossed entries (each the
        mirror of its recorded beam swap), whose channels, colour flows and
        flavor rows are those of the expanded output's own directories."""
        folded, expanded = self._outputs('p p > w+ j', 'wj')
        fentries, eentries, crossed = self._check_folding(folded, expanded)
        self.assertEqual(len(self._pdirs(folded)), 1)
        self.assertEqual(len(crossed), 2)
        for entry in crossed:
            same = [e for e in eentries
                    if (e['incoming'], e['outgoing']) ==
                    (entry['incoming'], entry['outgoing'])]
            self.assertEqual(len(same), 1)
            other = same[0]
            self.assertEqual(self._rows([entry]), self._rows([other]))
            self.assertEqual(entry['color_flows'], other['color_flows'])
            self.assertEqual(sorted(map(tuple, entry['helicities'])),
                             sorted(map(tuple, other['helicities'])))
            topology = lambda e: [(c['propagators'], c['vertices'],
                                   c['on_shell_propagators'],
                                   [(d['permutation'], d['propagator_pdgs'])
                                    for d in c['diagrams']])
                                  for c in e['channels']]
            self.assertEqual(topology(entry), topology(other))

    def test_ungrouped_beam_swaps_keep_entries_of_their_own(self):
        """set group_subprocesses False: the expanded output collects no
        mirror, so the beam swap of a crossed process (and the base's own,
        Q g > w+ Q, recorded as a crossing) is an entry of its own there. The
        folded output used to pair them into one mirrored entry all the same:
        the same partonic processes, laid out as no expanded output is."""
        folded, expanded = self._outputs(
            'p p > w+ j', 'wj_nogroup',
            setup=('set group_subprocesses False',))
        fentries, _, _ = self._check_folding(folded, expanded)
        self.assertFalse([flavor for entry in fentries
                          for flavor in entry['flavors'] if flavor['mirror']])

    def test_identical_final_antiquarks_keep_their_slot_order(self):
        """p p > w+ j j: q~ q~ > w+ q~ q~ has identical final antiquarks, and
        a crossing-table row serving the right physical process with those two
        swapped makes the base fill the amp2 of diagrams the crossed process
        does not have in that order -- the selected diagram then has no
        channel and the LHE writer gives up ("Diagram index out of range").
        The rows are served as the crossed process lists them instead."""
        folded, expanded = self._outputs('p p > w+ j j', 'wjj')
        self._check_folding(folded, expanded)

    def test_electroweak_dijet(self):
        """p p > j j QCD=0: the w+ and w- exchanges are the same propagators
        up to the particle; the crossed diagrams must find their own."""
        folded, expanded = self._outputs('p p > j j QCD=0', 'jjew')
        self._check_folding(folded, expanded)

    def test_crossing_moving_a_z_into_a_gluon_slot_is_folded(self):
        """q x > q x (x = g z): every crossing recorded on u g > u z is folded,
        also those moving the z into the gluon slot (u z > u g, u~ z > u~ g).
        They used to be expanded: the entry reused the BASE helicity table,
        which has no row for such a crossing. A crossed entry now ships the
        crossed process's own table and the backend reports that process's own
        code (TestCrossingHelicityConvention), so each folded entry's table is
        the one the expanded output writes for the same process."""
        folded, expanded = self._outputs('q x > q x QED=1 QCD=1', 'zg',
                                         defines=('q = u u~', 'x = g z'))
        fentries = self._entries(folded)
        eentries = self._entries(expanded)
        self.assertEqual(self._rows(fentries), self._rows(eentries))
        crossed = [e for e in fentries if e.get('crossing')]
        key = lambda e: (tuple(e['incoming']), tuple(e['outgoing']))
        self.assertEqual(sorted(key(e) for e in crossed),
                         [((-81, 21), (-81, 23)), ((-81, 23), (-81, 21)),
                          ((81, 23), (81, 21))])
        own = dict((key(e), e['helicities']) for e in eentries)
        for entry in crossed:
            self.assertEqual(entry['helicities'], own[key(entry)], key(entry))
        self.assertEqual(len(self._pdirs(folded)), 1)

    def test_decay_chain_crossings_are_expanded(self):
        """p p > z j, z > e+ e-: the crossings recorded inside a decay chain
        are expanded (folds_decay_chain_crossings)."""
        folded, expanded = self._outputs('p p > z j, z > e+ e-', 'zdecay')
        fentries = self._entries(folded)
        self.assertFalse([e for e in fentries if e.get('crossing')])
        self.assertEqual(self._rows(fentries),
                         self._rows(self._entries(expanded)))
        self.assertEqual(self._pdirs(folded), self._pdirs(expanded))

    def _session(self, *lines):
        cmd = cmd_interface.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False')
        cmd.exec_cmd('set apply_flavor_grouping True')
        cmd.exec_cmd('import model sm')
        for line in lines:
            cmd.exec_cmd(line)
        return cmd

    def test_crossing_turned_off_on_a_later_line_expands(self):
        """`add process ... --use_crossing=False` turns the crossing machinery
        off for the whole definition, while the `generate ... --use_crossing=
        True` line before it has recorded its crossings: they must come back
        as subprocesses of their own. The interface used to fold them (it only
        asked the output line) and the exporter, without the machinery, wrote
        them nowhere: q q~ > w+ g and g q~ > w+ q~ were missing from
        subprocesses.json and the cross section came out 30% low."""
        cmd = self._session('generate p p > w+ j --use_crossing=True',
                            'add process p p > w- j --use_crossing=False')
        self.assertTrue(cmd._has_recorded_crossings(cmd._curr_amps))
        self.assertFalse(cmd.output_uses_crossing())
        mixed = pjoin(self.tmpdir, 'mixed')
        cmd.exec_cmd('output mg7 %s -f --noeps=True' % mixed)
        reference = pjoin(self.tmpdir, 'mixed_reference')
        self._session('generate p p > w+ j --use_crossing=False',
                      'add process p p > w- j --use_crossing=False').exec_cmd(
            'output mg7 %s -f --noeps=True' % reference)
        entries = self._entries(mixed)
        self.assertFalse([e for e in entries if e.get('crossing')])
        self.assertEqual(self._rows(entries),
                         self._rows(self._entries(reference)))
        self.assertEqual(self._pdirs(mixed), self._pdirs(reference))

    def test_restricted_multiparticle_records_servable_crossings(self):
        """define p = g u d u~ d~; p p > j j: the generator matched crossings
        by the merged ids alone and recorded q q~ > q q~ on the u/d-only
        q q > q q, whose rows cannot give its u u~ > c c~ -- output mg7 then
        stopped half-written. A crossing is recorded only when each of its
        legs keeps within the flavors of the base leg it is paired with: the
        gluon-initiated crossings of g g > q q~ still fold, q q~ > q q~ and
        q~ q~ > q~ q~ are matrix elements of their own, and the output
        integrates what the expanded one does."""
        folded, expanded = self._outputs('p p > j j', 'restricted',
                                         defines=('p = g u d u~ d~',))
        self._check_folding(folded, expanded)
        crossed = set((tuple(e['incoming']), tuple(e['outgoing']))
                      for e in self._entries(folded) if e.get('crossing'))
        self.assertIn(((21, 81), (21, 81)), crossed)
        self.assertNotIn(((81, -81), (81, -81)), crossed)
        self.assertTrue([d for d in self._pdirs(folded)
                         if d.endswith('_QQx_QQx')], self._pdirs(folded))

    def test_reused_crossing_owns_its_records(self):
        """define p = g u d u~ d~; p p > w+ j: g Qx > w+ Qx cannot be recorded
        on g Q > w+ Q (its restricted legs), so it reuses the diagrams through
        cross_amplitude -- a shallow copy that kept the base's crossed_processes
        list. The two amplitudes then shared the base's records (with the
        base's leg order) and output mg7 / madevent died in HelasMatrixElement.
        Each amplitude must own its records, and the folded output must
        integrate what the expanded one does."""
        cmd = self._session('define p = g u d u~ d~',
                            'generate p p > w+ j --use_crossing=True')
        lists = [amp.get('crossed_processes') for amp in cmd._curr_amps]
        self.assertEqual(len(set(map(id, lists))), len(lists),
                         'amplitudes share a crossed_processes list')
        folded = pjoin(self.tmpdir, 'reused_folded')
        expanded = pjoin(self.tmpdir, 'reused_expanded')
        cmd.exec_cmd('output mg7 %s -f --noeps=True' % folded)
        cmd.exec_cmd('output mg7 %s -f --noeps=True --use_crossing=False'
                     % expanded)
        self._check_folding(folded, expanded)
        cmd.exec_cmd('output madevent %s -f --noeps=True'
                     % pjoin(self.tmpdir, 'reused_madevent'))

    # ------------------------------------------------------------------
    # runtime
    # ------------------------------------------------------------------
    def _generate_events(self, outdir, datadir, events=20000):
        """Run bin/generate_events with a fixed seed, no systematics; returns
        (returncode, info.json 'process' or None, log text)."""
        import glob
        from tests.acceptance_tests.test_cmd_madevent import _set_toml_key
        toml = pjoin(outdir, 'Cards', 'run_card.toml')
        with open(toml) as fsock:
            text = fsock.read()
        text = re.sub(r'(?m)^seed = -?\d+', 'seed = 4242', text)
        text = re.sub(r'(?m)^events = \d+', 'events = %d' % events, text)
        text = _set_toml_key(text, 'systematics', 'enable', 'false')
        with open(toml, 'w') as fsock:
            fsock.write(text)
        env = dict(os.environ, LHAPDF_DATA_PATH=datadir)
        log = pjoin(outdir, 'generate_events.log')
        with open(log, 'w') as fsock:
            ret = subprocess.call(
                [sys.executable, pjoin(outdir, 'bin', 'generate_events'), '-f'],
                cwd=outdir, env=env, stdout=fsock, stderr=subprocess.STDOUT)
        with open(log) as fsock:
            text = fsock.read()
        infos = sorted(glob.glob(pjoin(outdir, 'Events', '*', 'info.json')))
        info = json.load(open(infos[-1]))['process'] if infos else None
        return ret, info, text

    def _datadir(self):
        from tests.acceptance_tests.test_cmd_madevent import \
            _mg7_datadir_or_skip
        return _mg7_datadir_or_skip(self)

    def test_folded_cross_section_matches_expanded(self):
        """p p > w+ j: the folded and the expanded outputs integrate the same
        subprocesses with the same channels, so with the same seed they agree
        (to the last digits, in fact); asked here within the errors."""
        datadir = self._datadir()
        folded, expanded = self._outputs('p p > w+ j', 'wjxs')
        results = []
        for outdir in (folded, expanded):
            ret, info, log = self._generate_events(outdir, datadir)
            self.assertEqual(ret, 0, log[-2000:])
            self.assertTrue(info, 'no info.json in %s' % outdir)
            self.assertNotIn('Traceback', log)
            results.append((float(info['mean']), float(info['error'])))
        (x1, e1), (x2, e2) = results
        self.assertLess(abs(x1 - x2), 4 * math.sqrt(e1 ** 2 + e2 ** 2) + 1e-12,
                        'folded %s +- %s vs expanded %s +- %s'
                        % (x1, e1, x2, e2))


class TestCrossingProductDefault(unittest.TestCase):
    """The ONE test of the shipped default: crossing is on unless the user asks
    otherwise. Every other test of this suite pins its choice (_pin_crossing),
    so none of them can tell."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='cross_default_')

    def tearDown(self):
        if os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _subprocesses(self, options, name):
        cmd = cmd_interface.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False')
        cmd.exec_cmd('import model sm')
        cmd.exec_cmd(('generate p p > j j QCD=0 %s' % options).strip())
        out = pjoin(self.tmpdir, name)
        cmd.exec_cmd('output standalone_fortran %s -f' % out)
        return cmd, sorted(d for d in os.listdir(pjoin(out, 'SubProcesses'))
                           if d.startswith('P'))

    def test_crossing_is_on_by_default(self):
        cmd, bare = self._subprocesses('', 'bare')
        self.assertTrue(cmd._use_crossing)
        self.assertTrue(cmd.output_uses_crossing())
        _, on = self._subprocesses('--use_crossing=True', 'on')
        _, off = self._subprocesses('--use_crossing=False', 'off')
        self.assertEqual(bare, on)
        # guard the guard: the two choices really differ for this process
        self.assertLess(len(on), len(off), 'p p > j j QCD=0 folded nothing')

    def test_madevent_tags_the_beams_a_crossing_moves(self):
        """A default madevent e- p > e- j shares matrix elements across
        crossings of the proton side only. The beam-polarisation / EVA refusal
        (check_card_consistency) reads per-beam tags, so a polarised electron
        beam -- standard for DIS -- is not refused: only beam 2 is tagged."""
        cmd = cmd_interface.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False')
        cmd.exec_cmd('import model sm')
        cmd.exec_cmd('generate e- p > e- j')
        out = pjoin(self.tmpdir, 'ep')
        cmd.exec_cmd('output madevent %s -f --noeps=True' % out)
        import madgraph.various.banner as banner_mod
        lim = banner_mod.ProcCharacteristic(
            pjoin(out, 'SubProcesses', 'proc_characteristics'))['limitations']
        self.assertIn('crossing', lim)
        self.assertIn('crossing_beams', lim)
        self.assertIn('crossing_moves_beam2', lim)
        self.assertNotIn('crossing_moves_beam1', lim)


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
            cmd.run_cmd('generate %s --use_crossing=True' % proc)
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

    @staticmethod
    def _class_reps(me):
        """Signed PDGs of each flavor class's representative (members[0], the
        row the madevent FLAVOR table is built from), in class order."""
        _classes, class_pdgs = me.get_external_flavors_with_iden(
            return_pdgs=True)
        return [tuple(members[0]) for members in class_pdgs]

    def test_partition_pp_jj(self):
        """Every routed flavor must be reproduced EXACTLY, slot by slot, by the
        base row its FLAV_IDX names crossed through the table row it names:
        the dependent passes its momenta in its own leg order. That pins the
        whole index -- the row K, the base flavor class (the representative,
        not the ordinal row of the physical flavor table, which names another
        process from three merged flavors on) and the direction the row is
        read in. (The general, non-involution case needs several quark
        flavors: TestCrossingRoutesFinalLegReorder.)"""
        import madgraph.iolibs.crossing_table as crossing_table
        groups, exp = self._groups('p p > j j')
        eliminated_any = False
        for g in groups:
            mes = g.get('matrix_elements')
            bases, routing = exp.partition_crossing_classes(mes, commit=True)
            self.assertEqual(len(routing), len(mes))
            for i in range(len(mes)):
                self.assertTrue(routing[i], 'a module with no flavors')
                reps = self._class_reps(mes[i])
                self.assertEqual(len(routing[i]), len(reps))
                for flav0, (b, iflav) in enumerate(routing[i]):
                    self.assertIn(b, bases)            # routes to a real base
                    self.assertGreaterEqual(iflav, 1)  # 1-based FLAV_IDX
                    if i in bases:
                        self.assertEqual((b, iflav), (i, flav0 + 1))
                        continue
                    # an eliminated module never routes back to itself
                    self.assertNotEqual(b, i)
                    nflav_b = len(self._class_reps(mes[b]))
                    K, bflav = divmod(iflav - 1, nflav_b)
                    row = exp.madevent_crossing_table(mes[b])[K]
                    self.assertIn(-1, row.SD, 'not a genuine crossing')
                    anti = crossing_table.make_anti(
                        mes[b].get('processes')[0].get('model'))
                    self.assertEqual(
                        row.crossed(self._class_reps(mes[b])[bflav], anti),
                        reps[flav0],
                        '%s class %d is routed to a row that does not '
                        'reproduce it' % (mes[i].get('processes')[0]
                                          .shell_string(), flav0))
            if len(bases) < len(mes):
                eliminated_any = True
        self.assertTrue(eliminated_any,
                        'no module was eliminated by crossing in p p > j j')

    def test_partition_is_pure_without_commit(self):
        """Asking whether a group routes (commit=False, what the cross-group
        routing does group by group) must leave every table as it was."""
        groups, exp = self._groups('p p > j j')
        for g in groups:
            mes = g.get('matrix_elements')
            before = exp.partition_crossing_classes(mes)
            self.assertEqual([len(exp.madevent_crossing_table(me))
                              for me in mes], [1] * len(mes))
            self.assertEqual(exp.partition_crossing_classes(mes), before)


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
            bases, routing = exp.partition_crossing_classes(mes, commit=True)
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
            perm = exp.madevent_crossing_table(base)[cross].D
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
            cmd.exec_cmd('generate %s --use_crossing=True' % proc, printcmd=False)
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


class TestCrossingRoutesFinalLegReorder(unittest.TestCase):
    """The flavour-changing annihilation ``q q~ > q' q~'`` of ``Q Q~ > Q Q~``
    routes off ``Q Q > Q Q``.

    The crossing that reaches it delivers the two light final legs the other
    way round from the module's own leg order. The former
    I*(NEXTERNAL+1)+J code could not reorder final legs, so that one class kept
    the whole module compiled (and a generation-time split, MG_SPLIT_CROSSING,
    was needed to free it). A crossing-table row is any permutation: the class
    routes through a 3-cycle, and the module becomes a router."""

    def _groups(self, proc):
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
        for g in groups:
            g.generate_matrix_elements()
        return groups, export_v4.ProcessExporterFortranMEGroup()

    def test_qqx_module_routes_every_class(self):
        """Every class of Q Q~ > Q Q~ routes, each reproduced EXACTLY by the
        base row its FLAV_IDX names crossed through its table row -- with the
        default multi-flavor j, where it counts: one routed row is a 3-cycle,
        and one class of the module is not the ordinal row of its physical
        flavor table (a representative taken by position would name another
        process)."""
        import madgraph.iolibs.crossing_table as crossing_table
        groups, exp = self._groups('p p > j j')
        found = misaligned = False
        for g in groups:
            mes = g.get('matrix_elements')
            names = [m.get('processes')[0].shell_string(print_id=False)
                     for m in mes]
            if 'QQx_QQx' not in names:
                continue
            found = True
            i = names.index('QQx_QQx')
            bases, routing = exp.partition_crossing_classes(mes, commit=True)
            self.assertNotIn(i, bases, 'Q Q~ > Q Q~ still keeps its own '
                             'matrix element: %s' % names)
            reps = TestCrossingPartition._class_reps(mes[i])
            physical = [pdg for _f, pdg in
                        exp.crossing_base_entries(mes[i], 'rows')]
            misaligned = any(rep != physical[c] for c, rep in enumerate(reps))
            rows = []
            for flav0, (b, iflav) in enumerate(routing[i]):
                base_reps = TestCrossingPartition._class_reps(mes[b])
                K, bflav = divmod(iflav - 1, len(base_reps))
                row = exp.madevent_crossing_table(mes[b])[K]
                rows.append(row)
                anti = crossing_table.make_anti(
                    mes[b].get('processes')[0].get('model'))
                self.assertEqual(row.crossed(base_reps[bflav], anti),
                                 reps[flav0],
                                 'class %d of Q Q~ > Q Q~ is routed to a row '
                                 'that does not reproduce it' % flav0)
            self.assertTrue(
                any(any(r.D[r.D[k]] != k for k in range(len(r.D)))
                    for r in rows),
                'no class of Q Q~ > Q Q~ needed a non-involution row; the '
                'fixture no longer covers the final-leg reorder')
        self.assertTrue(found, 'no Q Q~ > Q Q~ module in p p > j j')
        self.assertTrue(misaligned, 'every class of Q Q~ > Q Q~ is its ordinal '
                        'physical row: the representative check has no teeth')


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

    The quarks pin the other half: every one of them sits on the W line, so a
    (massless) quark is left-handed and an antiquark right-handed in EVERY
    event, incoming or outgoing, crossed or not. A crossed fermion changes
    between particle and antiparticle, which reverses its helicity-state
    order: relabelling it by copying the base's helicity DIGIT instead of its
    VALUE writes it with the wrong sign.

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
            f.write('generate p p > w+ j --use_crossing=True\n'
                    'output madevent %s -f\n'
                    'launch\n'
                    'set nevents 1000\n'
                    'set iseed 777\n' % outdir)
        subprocess.call([sys.executable, pjoin(MG5DIR, 'bin', 'madgraph'), card])

        lhe = pjoin(outdir, 'Events', 'run_01', 'unweighted_events.lhe.gz')
        self.assertTrue(os.path.isfile(lhe),
                        'madevent produced no LHE file (%s)' % lhe)

        counts = {-1: 0, 0: 0, 1: 0}
        nevt = 0
        wrong_quark = []
        for event in lhe_parser.EventFile(lhe):
            for part in event:
                if 1 <= abs(part.pid) <= 4 and part.status in (-1, 1):
                    want = -1 if part.pid > 0 else 1
                    if int(round(part.helicity)) != want:
                        wrong_quark.append((part.pid, part.status,
                                            part.helicity))
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

        self.assertFalse(wrong_quark,
                         '%d quark(s) with the wrong helicity for a W vertex '
                         '(pid, status, helicity), e.g. %s'
                         % (len(wrong_quark), wrong_quark[:5]))
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
                        'set iseed 424242\n' % (_pin_crossing(options), outdir))
        subprocess.call([sys.executable, pjoin(MG5DIR, 'bin', 'madgraph'), card])
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
    router: they carry a ``crossgroup.mk`` (and their base a
    ``crossgroup_shared.dat``) instead of their own matrix element, i.e. their
    flavors are evaluated by ANOTHER group's matrix element under a crossing,
    over the helicity rows both groups' own surveys found. Nothing else in the suite
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
    # lines written before the generate line (multiparticle definitions)
    PRELUDE = ''
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
            fsock.write(self.PRELUDE)
            fsock.write('generate %s %s\n'
                        'output madevent %s -f\n'
                        'launch\n'
                        'set nevents %d\n'
                        'set iseed %d\n'
                        % (self.PROCESS, _pin_crossing(options), outdir,
                           self.NEVENTS, self.SEED))
        subprocess.call([sys.executable, pjoin(MG5DIR, 'bin', 'madgraph'), card])
        results = pjoin(outdir, 'SubProcesses', 'results.dat')
        self.assertTrue(
            os.path.isfile(results),
            'madevent produced no results for %s (%s)'
            % (options or 'the pinned (crossing on) build', results))
        with open(results) as fsock:
            # results.dat: cross-section, abs error, ... (in pb).
            fields = fsock.readline().split()
        return outdir, float(fields[0]), float(fields[1])

    @staticmethod
    def _routed_groups(outdir):
        """The subprocess groups served through a crossing: a within-group
        router (matrix<i>_router.f) or another group's matrix element
        (crossgroup.mk)."""
        subproc = pjoin(outdir, 'SubProcesses')
        routed = []
        for name in sorted(os.listdir(subproc)):
            pdir = pjoin(subproc, name)
            if not name.startswith('P') or not os.path.isdir(pdir):
                continue
            if any(re.match(r'matrix\d+_router\.f$', entry)
                   or entry == 'crossgroup.mk'
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


class TestMadeventMassiveLegCrossingXsec(TestMadeventInclusiveCrossingXsec):
    """The inclusive cross section of a crossing whose base has a massive leg,
    with helicity recycling (the default).

    `u b1 > c1 d1` (b1 = g u~, c1 = u z, d1 = z g): u u~ > z g is evaluated by
    the u g > u z group. The recycled optim of the base used to be baked over
    G_base U tau(G_base), the routed rows PREDICTED from the base's zeros. But
    G_base is measured in the u g frame, where four rows are exact zeros only
    because the z is massive (its helicity states mix under a boost), and their
    tau images are rows u u~ > z g needs: the routed process came out ~16% low
    -- 1802 +- 3.5 pb against 1914 +- 4.0 pb in total, with the z's helicity-0
    share 0.132 instead of 0.227. The optim now covers the rows the routed
    directory's own survey found (crossgroup_shared.dat), each crossing scanned
    for itself as madspace does: 1917 +- 3.5 pb.
    """

    PRELUDE = ('define b1 = g u~\n'
               'define c1 = u z\n'
               'define d1 = z g\n')
    PROCESS = 'u b1 > c1 d1 QED=1 QCD=1'
    NEVENTS = 2000
    SEED = 4242

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='cross_mev_uz_')

    def test_inclusive_crossing_xsec_matches(self):
        super().test_inclusive_crossing_xsec_matches()

    def test_shared_base_lists_its_routed_directory(self):
        """Run-free: the base u g > u z is marked as shared and names the
        directory routing through it, whose own helicity survey (u u~ > z g at
        its own kinematics) gen_ximprove merges into the base's recycled optim
        -- each crossing scanned for itself, as madspace does, rather than its
        rows predicted from the base's zeros."""
        from madgraph import MG5DIR
        outdir = pjoin(self.tmpdir, 'out')
        card = pjoin(self.tmpdir, 'cmd_out.txt')
        with open(card, 'w') as fsock:
            fsock.write(self.PRELUDE + 'generate %s --use_crossing=True\n'
                        'output madevent %s -f\n' % (self.PROCESS, outdir))
        subprocess.call([sys.executable, pjoin(MG5DIR, 'bin', 'madgraph'), card])
        routed = self._routed_groups(outdir)
        self.assertEqual(len(routed), 1, routed)
        subproc = pjoin(outdir, 'SubProcesses')
        bases = [d for d in os.listdir(subproc)
                 if os.path.exists(pjoin(subproc, d, 'crossgroup_shared.dat'))]
        self.assertEqual(len(bases), 1, bases)
        with open(pjoin(subproc, bases[0], 'crossgroup_shared.dat')) as fsock:
            lines = [line.split() for line in fsock if line.strip()]
        self.assertEqual(lines, [['1'] + routed])
        # the retired prediction tables are gone
        for name in ('crossgroup_helunion.dat', 'crossgroup_helclass.dat'):
            self.assertFalse(os.path.exists(pjoin(subproc, bases[0], name)))


class TestMadeventAmplitudeChunkCompiles(unittest.TestCase):
    """A madevent matrix element whose HELAS call sequence is split into
    amplitude chunk files (matrix<i>_origamp<k>.f) compiles with crossing off
    as well as on.

    The chunk routines used to take the crossing's NSF flags IC in every
    build, and the call in MATRIX passed IC -- which only a matrix element
    carrying the crossing machinery declares: with --use_crossing=False the
    chunked matrix element did not compile ("Symbol 'ic' has no IMPLICIT
    type"). Chunks appear only above amp_chunk_size statements, i.e. for large
    processes (p p > t t~ j j j, p p > 5j), so a small chunk size forces them
    on a small process here."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='cross_ampchunk_')

    def tearDown(self):
        if os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _compile_chunked(self, use_crossing):
        outdir = pjoin(self.tmpdir, 'out_%s' % use_crossing)
        cmd = cmd_interface.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False')
        cmd.exec_cmd('import model sm')
        cmd.exec_cmd('generate p p > w+ j --use_crossing=%s' % use_crossing)
        cmd.exec_cmd('output madevent %s -f --amp_chunk_size=2' % outdir)
        with open(os.devnull, 'w') as devnull:
            subprocess.call(['make'], cwd=pjoin(outdir, 'Source'),
                            stdout=devnull, stderr=devnull)
        flags = ['gfortran', '-O2', '-w', '-ffixed-line-length-132', '-I.',
                 '-I../../Source', '-I../../Source/MODEL',
                 '-I../../Source/DHELAS']
        chunked = 0
        for pdir in sorted(misc.glob('P*', pjoin(outdir, 'SubProcesses'))):
            chunks = misc.glob('matrix*_origamp*.f', pdir)
            if not chunks:
                continue
            chunked += 1
            for source in sorted(chunks) + misc.glob('matrix*_orig.f', pdir):
                if os.path.islink(source):
                    continue
                proc = subprocess.run(
                    flags + ['-c', os.path.basename(source), '-o',
                             os.devnull], cwd=pdir, capture_output=True,
                    text=True)
                self.assertEqual(proc.returncode, 0, '%s does not compile '
                                 '(--use_crossing=%s):\n%s'
                                 % (source, use_crossing, proc.stderr[-2000:]))
        self.assertTrue(chunked, 'no amplitude chunk file was written')

    def test_chunked_matrix_element_compiles_without_crossing(self):
        if not shutil.which('gfortran'):
            self.skipTest('no gfortran')
        self._compile_chunked(False)

    def test_chunked_matrix_element_compiles_with_crossing(self):
        if not shutil.which('gfortran'):
            self.skipTest('no gfortran')
        self._compile_chunked(True)


class TestCrossingHelicityConvention(unittest.TestCase):
    """Every backend labels the helicity of a crossed subprocess with the
    crossed process's OWN canonical code -- the code its expanded output uses.

    The backends evaluate a crossing on the BASE helicity rows (tau: momenta
    and NSF flags move into the base slots, the helicity slots do not), so each
    one needs a table turning that into the crossed process's code:

      * madevent: the routed subprocess relabels the base's selected code
        through its own state table (XDST/NXDST in auto_dsig<i>.f);
      * madmatrix (C++ cpu/simd/gpu): selected_hel_code encodes over the
        per-crossing state table (xhel_nhstate/xhel_states, row K);
      * mg7: the crossed subprocesses.json entry decodes the reported code with
        the helicity table it ships;
      * fortran standalone: SMATRIXHEL at an extended index takes that code
        (CROSS_HELCODE; checked at run time in
        test_massive_leg_crossing_after_base_training).

    They used to disagree: the C++ code was the BASE row whose per-slot values
    equal the crossed configuration, decoded by mg7 against the base table --
    which is why mg7 had to expand any crossing moving a leg into a slot with
    other helicity states. Run-free: on `u b1 > c1 d1` (b1 = g u~, c1 = u z,
    d1 = z g; u u~ > z g folded onto u g > u z, the z moved into a quark slot)
    the three tables must equal the expanded output's own helicity table, row
    for row (test_moved_leg_reports_its_own_helicity and
    TestMadeventMassiveLegCrossingXsec check the events)."""

    PRELUDE = ['define b1 = g u~', 'define c1 = u z', 'define d1 = z g']
    PROCESS = 'u b1 > c1 d1 QED=1 QCD=1'

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix='cross_helconv_')
        cmd = cmd_interface.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('set automatic_html_opening False')
        cmd.exec_cmd('import model sm')
        for line in cls.PRELUDE:
            cmd.exec_cmd(line)
        cmd.exec_cmd('generate %s --use_crossing=True' % cls.PROCESS)
        cls.out = {}
        for name, line in (('madevent', 'output madevent %s -f'),
                           ('mg7', 'output mg7 %s -f'),
                           ('cpp', 'output standalone %s -f'),
                           ('expanded',
                            'output standalone %s -f --use_crossing=False')):
            cls.out[name] = pjoin(cls.tmpdir, name)
            cmd.exec_cmd(line % cls.out[name])

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.tmpdir):
            shutil.rmtree(cls.tmpdir)

    def pdir(self, name, suffix):
        root = pjoin(self.out[name], 'SubProcesses')
        found = [pjoin(root, d) for d in sorted(os.listdir(root))
                 if d.startswith('P') and d.endswith(suffix)]
        self.assertEqual(len(found), 1, '%s %s: %s' % (name, suffix, found))
        return found[0]

    @staticmethod
    def product(nstates, states, maxhel, offset=0):
        """The mixed-radix table (first leg most significant) of a state
        table flattened leg by leg, maxhel entries per leg."""
        legs = [states[(offset + k) * maxhel:(offset + k) * maxhel + n]
                for k, n in enumerate(nstates)]
        return [tuple(row) for row in itertools.product(*legs)]

    def expanded_table(self):
        with open(pjoin(self.pdir('expanded', '_uux_zg'), 'ProcessData.h')) as f:
            thel = f.read().split('tHel')[1].split(';')[0]
        rows = [tuple(int(v) for v in row) for row in re.findall(
            r'\{\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\s*\}', thel)]
        self.assertEqual(len(rows), 24)
        return rows

    def test_madmatrix_reports_the_crossed_code(self):
        fold = self.pdir('cpp', '_ug_uz')
        with open(pjoin(fold, 'crossing_demo.dat')) as f:
            [fid] = [int(i) for i in f.read().split()]
        with open(pjoin(fold, 'ProcessTables.h')) as f:
            tables = f.read()

        def table(name):
            return [int(v) for v in re.search(
                r'%s\[[^\]]*\] = \{([^}]*)\}' % name, tables).group(1).split(',')]
        maxhel = int(re.search(r'xhel_maxhel = (\d+)', tables).group(1))
        nh, st = table('xhel_nhstate'), table('xhel_states')
        K = fid  # a single base flavor: the extended id is the row
        self.assertEqual(self.product(nh[4 * K:4 * K + 4], st, maxhel, 4 * K),
                         self.expanded_table())

    def test_mg7_folds_and_ships_the_crossed_table(self):
        root = pjoin(self.out['mg7'], 'SubProcesses')
        self.assertFalse(any(d.endswith('_uux_zg') for d in os.listdir(root)),
                         'output mg7 expanded the crossing that moves the z '
                         'into a quark slot')
        with open(pjoin(root, 'subprocesses.json')) as f:
            entries = json.load(f)
        crossed = [e for e in entries if e.get('crossing')]
        self.assertEqual(len(crossed), 1, crossed)
        self.assertEqual([tuple(r) for r in crossed[0]['helicities']],
                         self.expanded_table())

    def test_madevent_relabels_into_the_crossed_code(self):
        routed = self.pdir('madevent', '_qq_zg')
        self.assertTrue(os.path.exists(pjoin(routed, 'crossgroup.mk')),
                        'u u~ > z g is not evaluated through u g > u z')
        text = ''.join(open(f).read()
                       for f in sorted(misc.glob('auto_dsig*.f', routed)))
        states = [int(v) for v in
                  re.search(r'DATA XDST /([^/]*)/', text).group(1).split(',')]
        nstates = [int(v) for v in
                   re.search(r'DATA NXDST /([^/]*)/', text).group(1).split(',')]
        maxhel = len(states) // len(nstates)
        self.assertEqual(self.product(nstates, states, maxhel),
                         self.expanded_table())


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
            cmd.exec_cmd('generate %s --use_crossing=True' % proc, printcmd=False)
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
            cmd.exec_cmd('generate %s --use_crossing=True' % proc, printcmd=False)
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
            f.write('generate u u~ > u u~ --use_crossing=True\n'
                    'output madevent %s -f\n'
                    'launch\n'
                    'set nevents 2000\n'
                    'set iseed 909\n' % outdir)
        subprocess.call([sys.executable, pjoin(MG5DIR, 'bin', 'madgraph'), card])

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
                 'generate %s %s\n' % (self.PROCESS, _pin_crossing(options)),
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
        subprocess.call([sys.executable, pjoin(MG5DIR, 'bin', 'madgraph'), card])
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


class TestMadeventCrossingFinalLegReorder(unittest.TestCase):
    """madevent routes a flavor class whose crossing reorders the final legs.

    ``Q Q~ > Q Q~`` bundles three coupling classes and drops its own matrix
    element only if EVERY one of them routes. The flavour-changing annihilation
    ``q q~ > q' q~'`` is reached off ``Q Q > Q Q`` only with its two light final
    legs the other way round -- a 3-cycle, which the crossing table carries as
    one of its rows. (Under the former I*(NEXTERNAL+1)+J code it kept the module
    compiled, or needed a generation-time split of the class.)

    ``q q > q q`` with ``q = u d u~ d~`` rather than ``p p > j j``: same group,
    same class, no gluon subprocesses, so a generation takes seconds.

    What is pinned here:

    * the routing eliminates compiled matrix elements -- the one base serves
      every other subprocess of the group, each a router;
    * every router dispatches each of its flavors to an extended FLAV_IDX
      whose row exists in the base's compiled crossing table;
    * the group still lists the subprocesses and flavor combinations of the
      crossing-off build (the routers keep their own leshouche/PDF side).

    The numbers themselves are checked elsewhere: `check crossing q q > q q`
    per flavor, TestMadeventInclusiveCrossingXsec for the integral.

    The colour/helicity correctness of the routed events is NOT checked here --
    that needs event samples, and TestMadeventRouterColorSelection is where that
    kind of comparison lives.
    """

    DEFINE = 'define q = u d u~ d~'
    PROCESS = 'q q > q q'

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='cross_reorder_')

    def tearDown(self):
        if os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _generate(self, name, fmt='madevent', options=''):
        from madgraph import MG5DIR
        outdir = pjoin(self.tmpdir, name)
        card = pjoin(self.tmpdir, 'cmd_%s.txt' % name)
        with open(card, 'w') as fsock:
            fsock.writelines(['%s\n' % self.DEFINE,
                              'generate %s %s\n' % (self.PROCESS, _pin_crossing(options)),
                              'output %s %s -f -nojpeg\n' % (fmt, outdir)])
        subprocess.call([sys.executable, pjoin(MG5DIR, 'bin', 'madgraph'), card])
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
        directory covers, blind to the order the legs are listed in."""
        seen = {}
        for rows in cls._leshouche(pdir).values():
            for row in rows:
                key = (tuple(sorted(row[:nini])), tuple(sorted(row[nini:])))
                seen[key] = seen.get(key, 0) + 1
        return seen

    def test_routing_covers_the_flavors_and_frees_matrix_elements(self):
        plain = self._generate('plain', options='--use_crossing=False')
        routed = self._generate('routed')

        pdir_plain = pjoin(plain, 'SubProcesses', 'P1_qq_qq')
        pdir_routed = pjoin(routed, 'SubProcesses', 'P1_qq_qq')
        for pdir in (pdir_plain, pdir_routed):
            self.assertTrue(
                os.path.isfile(pjoin(pdir, 'leshouche.inc')),
                '%s has no leshouche.inc -- the generation did not finish'
                % pdir)

        # the same subprocesses, and fewer compiled matrix elements
        sub_plain = self._leshouche(pdir_plain)
        sub_routed = self._leshouche(pdir_routed)
        self.assertEqual(len(sub_routed), len(sub_plain))
        n_plain, r_plain = self._counts(pdir_plain)
        n_routed, r_routed = self._counts(pdir_routed)
        self.assertEqual(r_plain, 0,
                         '--use_crossing=False emitted %d router(s)' % r_plain)
        self.assertLess(n_routed, n_plain,
                        'the crossing compiles %d matrix element(s), no better '
                        'than the %d of --use_crossing=False'
                        % (n_routed, n_plain))
        self.assertEqual(r_routed, len(sub_routed) - n_routed,
                         'every subprocess of the routed group that is not a '
                         'compiled matrix element should be a router')
        # a single compiled Q Q > Q Q-type base serves the whole group: the
        # flavour-changing class did not keep Q Q~ > Q Q~ compiled
        self.assertEqual(n_routed, 1, 'Q Q~ > Q Q~ still keeps its own matrix '
                         'element (%d compiled)' % n_routed)

        want = self._physical(pdir_plain)
        got = self._physical(pdir_routed)
        self.assertEqual(got, want,
                         'the routed group does not list the flavor '
                         'combinations of the crossing-off build')

        # every routed call names a row the base was compiled with
        ncross = {}
        for name in os.listdir(pdir_routed):
            match = re.match(r'matrix(\d+)(_orig)?\.f$', name)
            if match:
                with open(pjoin(pdir_routed, name)) as fsock:
                    ncross[int(match.group(1))] = int(re.search(
                        r'PARAMETER\s*\(NCROSS=(\d+)\)', fsock.read()).group(1))
        with open(pjoin(pdir_routed, 'matrix1_orig.f'
                        if os.path.exists(pjoin(pdir_routed, 'matrix1_orig.f'))
                        else 'matrix1.f')) as fsock:
            nflav_base = int(re.search(r'PARAMETER\s*\(NFLAV=(\d+)\)',
                                       fsock.read()).group(1))
        calls = 0
        for name in os.listdir(pdir_routed):
            if not re.match(r'matrix\d+_router\.f$', name):
                continue
            with open(pjoin(pdir_routed, name)) as fsock:
                text = fsock.read()
            for base, iflav in re.findall(r'CALL SMATRIX(\d+)\(P,\s*(\d+),',
                                          text):
                self.assertIn(int(base), ncross, '%s routes to SMATRIX%s, '
                              'which is not compiled here' % (name, base))
                K = (int(iflav) - 1) // nflav_base
                self.assertTrue(1 <= K < ncross[int(base)],
                                '%s routes FLAV_IDX %s to row %d of a %d-row '
                                'table' % (name, iflav, K, ncross[int(base)]))
                calls += 1
        self.assertGreater(calls, 0, 'no routed call found in the routers')


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
                ['generate g g > u u~ %s\n' % _pin_crossing(options),
                 'add process u u~ > g g %s\n' % _pin_crossing(options),
                 'output madevent %s -f -nojpeg\n' % outdir,
                 'launch\n',
                 'set nevents %d\n' % self.NEVENTS,
                 'set iseed %d\n' % self.SEED,
                 # a broken local lhapdf kills the systematics step
                 'set use_syst False\n',
                 'set lpp2 -1\n'])
        subprocess.call([sys.executable, pjoin(MG5DIR, 'bin', 'madgraph'), card])
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
                     'crossgroup_shared.dat')
        self.assertTrue(
            os.path.exists(base),
            'the default build has no crossing base for g g > u u~ (no %s), so '
            'this test exercises no crossing at all' % os.path.basename(base))
        self.assertFalse(
            os.path.exists(pjoin(plain, 'SubProcesses', 'P1_gg_qq',
                                 'crossgroup_shared.dat')),
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
