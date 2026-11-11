################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
################################################################################
"""Radiative transitions between bound states as MadSpin decays.

    decay chic1 > Jpsi a, Jpsi > mu+ mu-
    decay Upsilon(2|3S11) > chib1(1|3P11) a, chib1(1|3P11) > Upsilon(1|3S11) a, Upsilon(1|3S11) > mu+ mu-

An electric-dipole (E1) or magnetic-dipole (M1) transition between two
colour-singlet bound states of the same constituents is non-perturbative: its
rate is the overlap of the two wave functions, which the branching ratio of the
param_card (a BR line of the parent's DECAY block) supplies. Its spin structure
is fixed at leading order in v:

  * E1 (L changes by one, the heavy-quark spin S is kept): the orbital part of
    the bound state couples to the photon through eps*.r, the spin goes over
    unchanged; chi_cJ > J/psi gamma, psi(2S) > chi_cJ gamma, h_c > eta_c gamma,
    chi_bJ(mP) > Upsilon(nS) gamma, Upsilon(nS) > chi_bJ(mP) gamma, ...
  * M1 (S changes by one, L kept): the spin couples to the magnetic field,
    (k x eps*).S, the orbital part goes over unchanged; J/psi > eta_c gamma,
    Upsilon(nS) > eta_b gamma, B_c* > B_c gamma, ...

Higher multipoles (M2, E3: of order k/m) are not included.

A colour-octet Fock state turns into its hadron in the same way, through a
soft gluon (chromo-E1 or chromo-M1, the photon replaced by a gluon that takes
the colour of the octet): 3S1[8] > chi_cJ g, 3PJ[8] > J/psi g (E1), 1S0[8] >
J/psi g, 3S1[8] > eta_c g (M1), or with L and S unchanged (3S1[8] > J/psi g,
1S0[8] > eta_c g: two E1 gluons at leading order in v) a gluon independent of
the spins, which keeps them. Integrated over the gluon, these are the
transfer rules of heavy-quark spin symmetry (onia_madspin.spin_transfer). The
octet is heavier than its hadron (0.2 GeV in sm_onia, as Onia:massSplit of
Pythia8), which gives the gluon its energy. A 3PJ[8] state of an S-wave
hadron is decayed in the uncoupled (L_z, S_z) basis, whose density has the
interference between the three J.

The amplitudes are written covariantly with the polarisation vectors of HELAS
(vxxxxx) that the bound-state matrix elements use, so that they live in the
same J_z basis as the production density matrix that MadSpin contracts them
with: a state of orbital L and spin S is the tensor

    sum_{L_z S_z} CG(L L_z; S S_z | J J_z) eps_L(L_z) x eps_S(S_z)

(the order of the template's CLEBSCH_GORDAN), eps of the incoming parent and
eps* of the outgoing daughter, of their own momenta and masses. With the photon
polarisation made gauge invariant in the parent's rest frame,
e = eps*_gamma - (eps*_gamma.P)/(k.P) k, the amplitudes are

    E1:  contract the orbital index (parent's or daughter's) with e, the two
         spin indices with each other;
    M1:  contract the spin index (parent's or daughter's) with
         w_rho = epsilon_{mu nu rho sigma} k^mu e^nu P^sigma, the two orbital
         indices with each other.

In the rest frame of the parent and for k -> 0 these are the non-relativistic
multipole amplitudes. The decay density of the parent, D(l, l') =
A(l) A(l')^*, sums the photon helicities and contracts the daughter's own decay
density (another radiative transition, or an ordinary decay evaluated by a
density library) over the daughter's J_z.
"""
from __future__ import absolute_import, division

import math

import numpy

GAMMA = 22
GLUON = 21
METRIC = numpy.diag([1., -1., -1., -1.])


def _levi_civita():
    import itertools
    eps = numpy.zeros((4, 4, 4, 4))
    for perm in itertools.permutations(range(4)):
        sign = 1
        p = list(perm)
        for i in range(4):
            while p[i] != i:
                j = p[i]
                p[i], p[j] = p[j], p[i]
                sign = -sign
        eps[perm] = sign
    return eps


LEVI_CIVITA = _levi_civita()


def dot(a, b):
    """Minkowski product (+,-,-,-), bilinear (no complex conjugation)."""
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def polarisation(p, mass, nhel, nsv):
    """The HELAS polarisation vector (E, x, y, z) of a vector of momentum p:
    eps(nhel) for nsv = -1 (incoming), eps*(nhel) for nsv = +1 (outgoing), as
    vxxxxx (aloha's Python port of the Fortran routine) gives them."""
    from aloha.template_files.wavefunctions import vxxxxx
    w = vxxxxx([float(x) for x in p], float(mass), int(nhel), int(nsv))
    return numpy.array([complex(w[2]), complex(w[3]), complex(w[4]), complex(w[5])])


def quantum_numbers(fock):
    """(S, L, J) of a Fock state."""
    return (fock.get('spin') - 1) // 2, fock.get('orbital'), fock.get('J')


def uncoupled_pair(code):
    """(L_z, S_z) of a helicity code -4..4 of an uncoupled P wave (the
    template's 3*(L_z+1)+(S_z+1)-4)."""
    return (code + 4) // 3 - 1, (code + 4) % 3 - 1


def helicities(fock, uncoupled=False):
    """The helicity codes of a bound state, in the order of its densities."""
    if uncoupled:
        return list(range(-4, 5))
    J = fock.get('J')
    return list(range(-J, J + 1))


def state_tensors(fock, p, mass, nsv, uncoupled=False):
    """{J_z: (tensor, labels)} of a bound state of momentum p: labels is the
    list of its vector indices among 'orb' (L = 1) and 'spin' (S = 1), in that
    order; the tensor is a 0-d array for L = S = 0. With uncoupled (a 3P
    state), the keys are the codes -4..4 of the (L_z, S_z) pairs and the
    tensors eps_L(L_z) x eps_S(S_z)."""
    from MadSpin.onia_madspin import clebsch_gordan
    S, L, J = quantum_numbers(fock)
    if L > 1 or S > 1:
        raise NotImplementedError('radiative transitions of %s (L = %d, S = %d)'
                                  % (fock.get('name'), L, S))
    eps = dict((m, polarisation(p, mass, m, nsv)) for m in (-1, 0, 1))
    out = {}
    if uncoupled:
        for code in range(-4, 5):
            Lz, Sz = uncoupled_pair(code)
            out[code] = (numpy.outer(eps[Lz], eps[Sz]), ['orb', 'spin'])
        return out
    for Jz in range(-J, J + 1):
        if L == 0 and S == 0:
            out[Jz] = (numpy.array(1. + 0j), [])
        elif L == 0:
            out[Jz] = (eps[Jz], ['spin'])
        elif S == 0:
            out[Jz] = (eps[Jz], ['orb'])
        else:
            t = numpy.zeros((4, 4), dtype=complex)
            for Lz in (-1, 0, 1):
                Sz = Jz - Lz
                if abs(Sz) <= 1:
                    cg = clebsch_gordan(1, Lz, 1, Sz, J, Jz)
                    if cg:
                        t += cg * numpy.outer(eps[Lz], eps[Sz])
            out[Jz] = (t, ['orb', 'spin'])
    return out


def transition_type(parent, daughter, boson=GAMMA):
    """The transition parent > daughter boson between bound states of the
    same constituents: 'E1' (L changed by one, spin kept) or 'M1' (spin
    changed by one, L kept), else None; a photon between colour singlets, a
    gluon from a colour octet to a colour singlet, which may also keep L and
    S ('kept': the gluon does not see the spins)."""
    if boson == GAMMA and (parent.get('color') != 1 or daughter.get('color') != 1):
        return None
    if boson == GLUON and (parent.get('color') != 8 or daughter.get('color') != 1):
        return None
    try:
        same = sorted(parent.get('particles')) == sorted(daughter.get('particles'))
    except Exception:
        same = True
    if not same:
        return None
    Si, Li, Ji = quantum_numbers(parent)
    Sf, Lf, Jf = quantum_numbers(daughter)
    if abs(Li - Lf) == 1 and Si == Sf:
        return 'E1'
    if Li == Lf and abs(Si - Sf) == 1:
        return 'M1'
    if boson == GLUON and Li == Lf and Si == Sf and (Li == 0 or Ji == Jf):
        return 'kept'
    return None


def _contract(tp, lp, td, ld, kind, e_low, w_low):
    subs = [''.join({'orb': 'a', 'spin': 'b'}[x] for x in lp),
            ''.join({'orb': 'c', 'spin': 'd'}[x] for x in ld)]
    ops = [tp, td]
    if kind == 'kept':
        # the spins and the orbital state unchanged, the gluon independent
        if 'orb' in lp:
            ops.append(METRIC)
            subs.append('ac')
        if 'spin' in lp:
            ops.append(METRIC)
            subs.append('bd')
    elif kind == 'E1':
        if 'orb' in lp:
            ops.append(e_low)
            subs.append('a')
        if 'orb' in ld:
            ops.append(e_low)
            subs.append('c')
        if 'spin' in lp:
            ops.append(METRIC)
            subs.append('bd')
    else:
        if 'spin' in lp:
            ops.append(w_low)
            subs.append('b')
        if 'spin' in ld:
            ops.append(w_low)
            subs.append('d')
        if 'orb' in lp:
            ops.append(METRIC)
            subs.append('ac')
    return numpy.einsum(','.join(subs) + '->', *ops)


def amplitudes(parent, daughter, P, M, k, pf, Mf, kind=None, uncoupled=False):
    """R[l, m, h] of parent(P, J_z = l) > daughter(pf, J_z = m) boson(k,
    helicity h = -1, +1), l and m from -J to J (l the codes -4..4 of the
    (L_z, S_z) pairs of an uncoupled parent), h in (-1, +1); P, k, pf are
    (E, px, py, pz). The boson's polarisations are those of a massless vector
    (photon or gluon; the colour is an overall factor)."""
    kind = kind or transition_type(parent, daughter)
    if kind not in ('E1', 'M1', 'kept'):
        raise ValueError('%s > %s is no E1 or M1 transition'
                         % (parent.get('name'), daughter.get('name')))
    P = numpy.asarray(P, dtype=float)
    k = numpy.asarray(k, dtype=float)
    tp = state_tensors(parent, P, M, -1, uncoupled)
    td = state_tensors(daughter, pf, Mf, +1)
    hp = helicities(parent, uncoupled)
    Jd = daughter.get('J')
    out = numpy.zeros((len(hp), 2 * Jd + 1, 2), dtype=complex)
    kP = dot(k, P)
    for ih, h in enumerate((-1, 1)):
        eg = polarisation(k, 0., h, +1)
        e = eg - dot(eg, P) / kP * k
        e_low = METRIC @ e
        w_low = numpy.einsum('mnrs,m,n,s->r', LEVI_CIVITA, k, e, P) if kind == 'M1' else None
        for a, l in enumerate(hp):
            for b, m in enumerate(range(-Jd, Jd + 1)):
                out[a, b, ih] = _contract(tp[l][0], tp[l][1], td[m][0], td[m][1], kind, e_low, w_low)
    if kind == 'kept':
        # a gluon independent of the spins: its two helicities share the rate
        out /= math.sqrt(2.)
    return out


def transfer(R, rho_daughter):
    """D(l, l') = sum_h sum_{m m'} R[l, m, h] R[l', m', h]^* rho_daughter[m, m']:
    the decay density of the parent from that of the daughter."""
    return numpy.einsum('amh,bnh,mn->ab', R, R.conj(), rho_daughter)


def daughter_density(R, rho_parent):
    """The (unnormalised) production density of the daughter for a parent of
    production density rho_parent (MadSpin's convention: rho(l, l') = A(l)
    A(l')^* with the outgoing parent's eps*): rho_f(m, m') = sum_h sum_{l l'}
    rho_parent[l, l'] R[l, m, h] R[l', m', h]^*, so that sum rho_P D_parent =
    sum rho_f D_f."""
    return numpy.einsum('ab,amh,bnh->mn', rho_parent, R, R.conj())


# --------------------------------------------------------------- kinematics --
def two_body(M, m1, m2, direction):
    """Momenta of M (at rest) > m1 + m2, m1 along the unit vector direction."""
    q = math.sqrt(max((M * M - (m1 + m2) ** 2) * (M * M - (m1 - m2) ** 2), 0.)) / (2 * M)
    d = numpy.asarray(direction, dtype=float)
    p1 = numpy.concatenate([[math.sqrt(q * q + m1 * m1)], q * d])
    p2 = numpy.concatenate([[math.sqrt(q * q + m2 * m2)], -q * d])
    return p1, p2


def boost(p, q):
    """p boosted by the velocity of q (from the rest frame of q to the frame
    where q has its momentum)."""
    p = numpy.asarray(p, dtype=float)
    m = math.sqrt(max(dot(q, q), 0.))
    bx = numpy.asarray(q[1:], dtype=float) / m
    gamma = q[0] / m
    bp = float(numpy.dot(bx, p[1:]))
    E = gamma * p[0] + bp
    vec = p[1:] + bx * (p[0] + bp / (gamma + 1.))
    return numpy.concatenate([[E], vec])


def random_direction(rng):
    c = 2 * rng.random() - 1
    phi = 2 * math.pi * rng.random()
    s = math.sqrt(max(1 - c * c, 0.))
    return numpy.array([s * math.cos(phi), s * math.sin(phi), c])


# ------------------------------------------------------------------ chains --
class Terminal(object):
    """The ordinary decay that ends a chain (e.g. 'Jpsi_1_3S11 > mu+ mu-'),
    evaluated by the density library of the radiative chains. products are the
    pdg codes of its final state; order, the leg order of the library
    (the decaying state first), is set when the library is built."""

    def __init__(self, fock, process, products):
        self.fock = fock
        self.process = process
        self.products = list(products)
        self.order = None

    def final_pdgs(self):
        return list(self.order[1:]) if self.order else list(self.products)

    def conjugate(self, conj):
        out = Terminal(self.fock, self.process, [conj(p) for p in self.products])
        out.order = tuple(conj(p) for p in self.order) if self.order else None
        out.particle = self
        return out


class Chain(object):
    """parent > daughter boson (E1, M1, or for the gluon of a colour octet
    'kept'), then sub: the daughter's own decay (a Chain, a Terminal) or None
    (the daughter is not decayed). boson is the photon of a radiative
    transition or the soft gluon of a colour octet; uncoupled, a 3P[8] parent
    in the (L_z, S_z) basis."""

    def __init__(self, parent, daughter, kind, sub, text='', boson=GAMMA, uncoupled=False):
        self.parent = parent
        self.daughter = daughter
        self.kind = kind
        self.sub = sub
        self.text = text
        self.boson = boson
        self.uncoupled = uncoupled
        self.parent_pdg = parent.get('pdg_code')
        self.daughter_pdg = daughter.get('pdg_code')

    def parent_helicities(self):
        return helicities(self.parent, self.uncoupled)

    def final_pdgs(self):
        """The final state of the chain in the order of its events: the boson,
        then the daughter's decay products (or the daughter itself)."""
        if self.sub is None:
            return [self.boson, self.daughter_pdg]
        return [self.boson] + self.sub.final_pdgs()

    def steps(self):
        out, c = [], self
        while isinstance(c, Chain):
            out.append(c)
            c = c.sub
        return out

    def terminal(self):
        return self.steps()[-1].sub

    def with_parent_pdg(self, pdg, conj=None):
        """The chain for an event whose decaying particle has pdg code pdg (a
        colour octet decaying as its hadron, or the antiparticle with conj)."""
        import copy
        out = copy.copy(self)
        out.parent_pdg = pdg
        if conj is not None:
            sub = self.sub
            if isinstance(sub, Chain):
                out.sub = sub.with_parent_pdg(conj(sub.parent_pdg), conj)
            elif isinstance(sub, Terminal):
                out.sub = sub.conjugate(conj)
            out.daughter_pdg = conj(self.daughter_pdg)
        return out

    def order(self):
        """The leg order of the chain's events: (parent,) + final_pdgs."""
        return (self.parent_pdg,) + tuple(self.final_pdgs())

    def tag(self):
        return ((self.parent_pdg,), tuple(sorted(self.final_pdgs())))


def parse_chain(registry, line, name2pdg):
    """The Chain of a decay line whose first step is a radiative transition
    between bound states ('chic1 > Jpsi a, Jpsi > mu+ mu-'), else None. Raises
    ValueError for a line that starts radiatively but cannot be used."""
    steps = []
    for part in line.split(','):
        part = part.strip()
        if not part:
            continue
        if part.count('>') != 1:
            return None
        lhs, rhs = part.split('>')
        steps.append((lhs.strip(), rhs.split()))
    if not steps:
        return None

    def onium(name):
        fock = registry.fock_of_name(name)
        return registry.singlet_of(fock) if fock is not None else None

    def radiative_pair(products):
        if len(products) != 2 or products.count('a') != 1:
            return None
        other = [p for p in products if p != 'a'][0]
        return onium(other)

    parent = onium(steps[0][0])
    if parent is None or radiative_pair(steps[0][1]) is None:
        return None
    later = {}
    for lhs, rhs in steps[1:]:
        f = onium(lhs)
        if f is None:
            raise ValueError("MadSpin: in '%s', '%s' is no bound state of the chain" % (line, lhs))
        if f.get('pdg_code') in later:
            raise ValueError("MadSpin: in '%s', %s decays twice" % (line, lhs))
        later[f.get('pdg_code')] = (lhs, rhs)

    from MadSpin.onia_madspin import alias

    def build(fock, products):
        daughter = radiative_pair(products)
        if daughter is None:
            return None
        kind = transition_type(fock, daughter)
        if kind is None:
            raise ValueError(
                "MadSpin: %s > %s a is neither an E1 transition (L changed by one, spin kept) nor "
                "an M1 transition (spin changed by one, L kept) between bound states of the same "
                "constituents" % (fock.get('name'), daughter.get('name')))
        sub = None
        step = later.pop(daughter.get('pdg_code'), None)
        if step is not None:
            sub = build(daughter, step[1])
            if sub is None:
                names = step[1]
                bad = [n for n in names if n not in name2pdg]
                if bad:
                    raise ValueError(
                        "MadSpin: the decay of %s at the end of a radiative chain takes particles "
                        "of the model only, one final state per decay line (not %s)"
                        % (daughter.get('name'), ', '.join(bad)))
                sub = Terminal(daughter, '%s > %s' % (alias(daughter.get('name')), ' '.join(names)),
                               [name2pdg[n] for n in names])
        return Chain(fock, daughter, kind, sub, line)

    chain = build(parent, steps[0][1])
    if later:
        raise ValueError("MadSpin: in '%s', the decay of %s is not part of the chain"
                         % (line, ', '.join(v[0] for v in later.values())))
    return chain


def ambiguous(chains, masses, tolerance=1e-6):
    """Pairs of chains with the same legs whose intermediate states have the
    same masses: their decay events cannot be told apart."""
    out = []
    for i, a in enumerate(chains):
        for b in chains[i + 1:]:
            if a.order() != b.order() or a is b:
                continue
            ma = [masses.get(abs(s.daughter.get('pdg_code'))) for s in a.steps()]
            mb = [masses.get(abs(s.daughter.get('pdg_code'))) for s in b.steps()]
            if len(ma) == len(mb) and all(x is not None and y is not None and
                                          abs(x - y) <= tolerance * max(x, y) for x, y in zip(ma, mb)):
                out.append((a, b))
    return out


def branching_ratio(chain, decay_block, defaults=None, used=None, lo_widths=None):
    """The product of the BR lines of the chain's steps in the DECAY blocks of
    the param_card; a step without a BR line takes the default branching
    fraction (defaults: {pdg: (width, {sorted products: fraction})}, the PDG
    values of madgraph/models/onia_decays.py or the model's own), noted in
    used as (what, fraction); raises ValueError naming a line that neither
    has. The soft gluon of a colour octet has branching ratio 1; the ordinary
    decay that ends the chain of a colour octet, without a BR line, takes its
    LO partial width (lo_widths: {process: width}) over the total width, as
    MadSpin does for the decays of the colour singlet."""
    def line(pdg, ids, what):
        table = decay_block.decay_table.get(abs(pdg))
        if table:
            for order in (sorted(ids), list(ids)):
                try:
                    return table.get(tuple([len(order)] + list(order))).value
                except KeyError:
                    continue
        if defaults:
            # the table of the particle; that of the antiparticle conjugated
            entry = defaults.get(pdg)
            key = tuple(sorted(ids))
            if entry is None and defaults.get(-pdg) is not None:
                entry = defaults.get(-pdg)
                key = tuple(sorted(-i if i not in (GAMMA,) else i for i in ids))
            if entry is not None and key in entry[1]:
                if used is not None:
                    used.append((what, entry[1][key]))
                return entry[1][key]
        raise ValueError(
            "MadSpin: no branching ratio for %s, in the param_card or the PDG values. A radiative "
            "transition is non-perturbative: give its BR line (and those of the decays that "
            "follow) in the DECAY block, e.g.\n   DECAY %d <total width>\n      <BR>  %d  %s" % (
                what, abs(pdg), len(ids), '  '.join(str(i) for i in sorted(ids))))
    br = 1.0
    for step in chain.steps():
        if step.boson == GLUON:
            continue
        br *= line(step.parent.get('pdg_code'), [step.daughter.get('pdg_code'), GAMMA],
                   '%s > %s a' % (step.parent.get('name'), step.daughter.get('name')))
    term = chain.terminal()
    if isinstance(term, Terminal):
        pdg = term.fock.get('pdg_code')
        if lo_widths is not None and term.process in lo_widths:
            table = decay_block.decay_table.get(abs(pdg))
            explicit = None
            if table:
                for order in (sorted(term.products), list(term.products)):
                    try:
                        explicit = table.get(tuple([len(order)] + list(order))).value
                        break
                    except KeyError:
                        continue
            if explicit is not None:
                br *= explicit
            else:
                br *= lo_widths[term.process] / decay_block.get((abs(pdg),)).value
        else:
            br *= line(pdg, term.products, term.process)
    return br


# ---------------------------------------------------------------- densities --
def unpack(packed, n):
    rho = numpy.zeros((n, n), dtype=complex)
    k = 0
    for i in range(n):
        for j in range(i, n):
            rho[i, j] = packed[k]
            rho[j, i] = numpy.conj(packed[k])
            k += 1
    return rho


def pack(rho):
    n = rho.shape[0]
    return numpy.array([rho[i, j] for i in range(n) for j in range(i, n)])


class Evaluator(object):
    """The decay density of a chain from the momenta of its final state."""

    def __init__(self, library, masses):
        self.library = library
        self.masses = masses        # {abs(pdg): mass} of the bound states

    def mass(self, pdg):
        return self.masses[abs(pdg)]

    def terminal_density(self, term, pf, finals):
        order = term.particle.order if hasattr(term, 'particle') else term.order
        pool = list(finals)
        moms = [pf]
        for pdg in (term.order or order)[1:]:
            i = next(i for i, (q, _) in enumerate(pool) if q == pdg)
            moms.append(pool.pop(i)[1])
        J = term.fock.get('J')
        hels = list(range(-J, J + 1))
        P = numpy.asfortranarray(numpy.array(moms, dtype=float).T)
        packed = self.library.py_get_density(pdgs=list(order), procid=-1, p=P, pos=[1],
                                             allow_hel=hels, alphas=0.118,
                                             scale2=self.mass(term.fock.get('pdg_code')) ** 2)
        return unpack(packed, len(hels))

    def offshell(self, chain, P, finals):
        """How far the final state is from the chain's intermediate states:
        the sum over its steps of |(P - k)^2 - M_f^2| / M_f^2 with the best
        photon of each step. Decay lines with the same final state (e.g.
        psi(2S) > chi_cJ gamma, chi_cJ > J/psi gamma for J = 0, 1, 2) differ by
        the masses of their intermediate states only."""
        P = numpy.asarray(P, dtype=float)
        total = 0.
        for step in chain.steps():
            Mf = self.mass(step.daughter.get('pdg_code'))
            photons = [i for i, (q, _) in enumerate(finals) if q == step.boson]
            if not photons:
                return float('inf')
            best, off = None, None
            for i in photons:
                q = P - numpy.asarray(finals[i][1], dtype=float)
                d = abs(dot(q, q) - Mf * Mf) / (Mf * Mf)
                if off is None or d < off:
                    best, off = i, d
            total += off
            P = P - numpy.asarray(finals[best][1], dtype=float)
            finals = [f for i, f in enumerate(finals) if i != best]
        return total

    def density(self, chain, P, finals):
        """D(l, l') of the chain's parent (momentum P) in its J_z -J..J, for
        the final state finals = [(pdg, momentum)] (any order). The photon of
        each step is the one that puts the daughter on its mass shell."""
        P = numpy.asarray(P, dtype=float)
        Mf = self.mass(chain.daughter.get('pdg_code'))
        photons = [i for i, (q, _) in enumerate(finals) if q == chain.boson]

        def offshell(i):
            q = P - numpy.asarray(finals[i][1], dtype=float)
            return abs(dot(q, q) - Mf * Mf)
        best = min(photons, key=offshell)
        k = numpy.asarray(finals[best][1], dtype=float)
        rest = [f for i, f in enumerate(finals) if i != best]
        pf = P - k
        if chain.sub is None:
            n = 2 * chain.daughter.get('J') + 1
            Df = numpy.eye(n, dtype=complex)
        elif isinstance(chain.sub, Chain):
            Df = self.density(chain.sub, pf, rest)
        else:
            Df = self.terminal_density(chain.sub, pf, rest)
        M = math.sqrt(max(dot(P, P), 0.))
        mf = math.sqrt(max(dot(pf, pf), 0.))
        A = amplitudes(chain.parent, chain.daughter, P, M, k, pf, mf, chain.kind, chain.uncoupled)
        return transfer(A, Df)


class RadiativeLibrary(object):
    """MadSpin's decay density library with the radiative chains added: the
    calls whose pdg codes are those of a chain are answered here, the others
    by the library (which may be None when MadSpin has no other decay)."""

    def __init__(self, base, evaluator, chains):
        self._base = base
        self._evaluator = evaluator
        # decay lines with the same final state share their legs: the
        # kinematics of the event (the masses of the intermediate states) tells
        # them apart
        self._chains = {}
        for c in chains:
            self._chains.setdefault(c.order(), []).append(c)

    def __getattr__(self, name):
        if self._base is None:
            raise AttributeError(name)
        return getattr(self._base, name)

    def chain(self, pdgs):
        """The chains whose legs are pdgs (None if there is none)."""
        return self._chains.get(tuple(int(x) for x in pdgs if int(x) != 0))

    def _density(self, chains, p):
        """(chain, D) for the momenta p (4 x legs): the chain whose
        intermediate states the final state fits, and its decay density."""
        P = numpy.asarray(p, dtype=float)
        finals = [(q, P[:, j + 1]) for j, q in enumerate(chains[0].final_pdgs())]
        chain = chains[0] if len(chains) == 1 else min(
            chains, key=lambda c: self._evaluator.offshell(c, P[:, 0], finals))
        return chain, self._evaluator.density(chain, P[:, 0], finals)

    def py_get_density(self, pdgs, procid, p, pos, allow_hel, alphas, scale2):
        chains = self.chain(pdgs)
        if chains is None:
            return self._base.py_get_density(pdgs=pdgs, procid=procid, p=p, pos=pos,
                                             allow_hel=allow_hel, alphas=alphas, scale2=scale2)
        chain, D = self._density(chains, p)
        hels = chain.parent_helicities()
        idx = [hels.index(int(h)) for h in allow_hel]
        return pack(D[numpy.ix_(idx, idx)])

    def py_get_density_batch(self, pdgs, procid, p, pos, allow_hel, alphas, scale2, nbatch, next):
        pdgs = numpy.asarray(pdgs)
        p = numpy.asarray(p)
        cols = [self.chain(pdgs[:, c]) for c in range(nbatch)]
        n = len(allow_hel) // len(pos)
        out = numpy.zeros((n * (n + 1) // 2, nbatch), dtype=complex)
        base = [c for c in range(nbatch) if cols[c] is None]
        if base:
            sub = self._base.py_get_density_batch(
                pdgs=numpy.asfortranarray(pdgs[:, base]), procid=procid,
                p=numpy.asfortranarray(p[:, :, base]), pos=pos, allow_hel=allow_hel,
                alphas=numpy.asarray(alphas)[base], scale2=numpy.asarray(scale2)[base],
                nbatch=len(base), next=next)
            for i, c in enumerate(base):
                out[:, c] = sub[:, i]
        for c in range(nbatch):
            if cols[c] is not None:
                out[:, c] = self.py_get_density(pdgs[:, c], procid, p[:, :, c], pos, allow_hel,
                                                None, None)
        return out

    def smatrixhel(self, pdgs, procid, p, alphas, scale2, nhel):
        chains = self.chain(pdgs)
        if chains is None:
            return self._base.smatrixhel(pdgs, procid, p, alphas, scale2, nhel)
        chain, D = self._density(chains, p)
        return numpy.trace(D).real / D.shape[0]


# ------------------------------------------------------------- decay events --
def _colour_flow(pdgs, colour_of, start=500):
    """Colour-singlet flow (ICOLUP pairs) for the products of a colourless
    state: quarks, gluons and antiquarks joined in one line (or a closed loop
    of gluons); the tags start after start."""
    cols = [[0, 0] for _ in pdgs]
    tag = [start]

    def new():
        tag[0] += 1
        return tag[0]
    quarks = [i for i, q in enumerate(pdgs) if colour_of(q) == 3]
    antis = [i for i, q in enumerate(pdgs) if colour_of(q) == -3]
    gluons = [i for i, q in enumerate(pdgs) if colour_of(q) == 8]
    if not quarks and not antis:
        if gluons:
            first = new()
            c = first
            for n, i in enumerate(gluons):
                nxt = first if n == len(gluons) - 1 else new()
                cols[i] = [c, nxt]
                c = nxt
        return cols
    for n, (q, qb) in enumerate(zip(quarks, antis)):
        c = new()
        cols[q] = [c, 0]
        if n == 0:
            for i in gluons:
                nxt = new()
                cols[i] = [nxt, c]
                c = nxt
        cols[qb] = [0, c]
    return cols


class PoolGenerator(object):
    """Decay events of a radiative chain, the parent at rest, distributed as
    the spin-summed |M|^2 of the whole chain (sum_l D(l, l)) on its phase
    space: the photon direction and an unweighted decay of the daughter are
    drawn flat, then kept with probability Tr D / bound."""

    def __init__(self, evaluator, masses, rng, colour_of=lambda pdg: 1):
        self.evaluator = evaluator
        self.masses = masses        # {abs(pdg): mass} of every particle met
        self.rng = rng
        self.colour_of = colour_of

    def mass(self, pdg):
        return self.masses.get(abs(pdg), 0.)

    def _terminal(self, term, M):
        """Momenta of the terminal's products in the daughter's rest frame,
        flat in phase space; (moms, weight)."""
        pdgs = term.final_pdgs()
        ms = [self.mass(q) for q in pdgs]
        if len(pdgs) == 2:
            p1, p2 = two_body(M, ms[0], ms[1], random_direction(self.rng))
            return [p1, p2], 1.0
        P, lwt = _rambo(M, ms)
        moms = [numpy.array([P[(4, i)], P[(1, i)], P[(2, i)], P[(3, i)]]) for i in range(1, len(pdgs) + 1)]
        return moms, math.exp(lwt)

    def _candidate(self, chain, M):
        """[(pdg, status, mother, momentum)] in the parent's rest frame (the
        parent is entry 0) and the phase-space weight."""
        Mf = self.mass(chain.daughter_pdg)
        k, pf = two_body(M, 0., Mf, random_direction(self.rng))
        out = [(chain.parent_pdg, -1, -1, numpy.array([M, 0., 0., 0.])), (chain.boson, 1, 0, k)]
        wt = 1.0
        sub = chain.sub
        if sub is None:
            out.append((chain.daughter_pdg, 1, 0, pf))
            return out, wt
        out.append((chain.daughter_pdg, 2, 0, pf))
        d = len(out) - 1
        if isinstance(sub, Chain):
            inner, w = self._candidate(sub, Mf)
            wt *= w
            base = len(out) - 1      # inner entry 0 (the daughter) is out[d]
            for pdg, status, mother, p in inner[1:]:
                out.append((pdg, status, d if mother == 0 else mother + base, boost(p, pf)))
        else:
            moms, w = self._terminal(sub, Mf)
            wt *= w
            for pdg, p in zip(sub.final_pdgs(), moms):
                out.append((pdg, 1, d, boost(p, pf)))
        return out, wt

    def weight(self, chain, cand):
        finals = [(pdg, p) for pdg, status, mother, p in cand if status == 1]
        D = self.evaluator.density(chain, cand[0][3], finals)
        return numpy.trace(D).real

    def generate(self, chain, nb_event, M, scan=500):
        """nb_event events of the chain (lists as _candidate gives them)."""
        bound = 0.
        for _ in range(scan):
            cand, wt = self._candidate(chain, M)
            bound = max(bound, wt * self.weight(chain, cand))
        bound *= 1.3
        events, tried, over = [], 0, 0
        while len(events) < nb_event:
            cand, wt = self._candidate(chain, M)
            w = wt * self.weight(chain, cand)
            tried += 1
            if w > bound:
                over += 1
                bound = w * 1.1
            if self.rng.random() * bound < w:
                events.append(cand)
        self.efficiency = len(events) / float(max(tried, 1))
        self.overflows = over
        return events

    def lhe_event(self, cand, scale):
        pdgs = [c[0] for c in cand]
        cols = {}
        # a colour octet: its colour goes to the soft gluon it emits (entry 1);
        # MadSpin maps the parent's tags onto those of the production
        octet = len(cand) > 1 and cand[1][0] == GLUON and cand[1][2] == 0
        if octet:
            cols[0] = (501, 502)
            cols[1] = (501, 502)
        finals = [i for i, c in enumerate(cand) if c[1] == 1 and abs(self.colour_of(c[0])) > 1
                  and i not in cols]
        if finals:
            flow = _colour_flow([pdgs[i] for i in finals], self.colour_of, start=502 if octet else 500)
            cols.update(dict(zip(finals, flow)))
        lines = ['<event>', ' %d 0 1.0 %.8e -1.0 0.118' % (len(cand), scale)]
        for i, (pdg, status, mother, p) in enumerate(cand):
            m = math.sqrt(max(dot(p, p), 0.)) if status != 1 or abs(pdg) not in (GAMMA, GLUON) else 0.
            if status == 1 and abs(pdg) not in (GAMMA, GLUON):
                m = self.mass(pdg)
            c1, c2 = cols.get(i, (0, 0))
            lines.append(' %d %d %d %d %d %d %+.11e %+.11e %+.11e %.11e %.11e 0.0 9.0' % (
                pdg, status, mother + 1 if mother >= 0 else 0, mother + 1 if mother >= 0 else 0,
                c1, c2, p[1], p[2], p[3], p[0], m))
        lines.append('</event>')
        return '\n'.join(lines)

    def write(self, path, events, cross, scale):
        import gzip
        import os
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with gzip.open(path, 'wt') as f:
            f.write('<LesHouchesEvents version="3.0">\n<header>\n'
                    '<!-- radiative transitions of bound states: MadSpin/onia_radiative.py -->\n'
                    '</header>\n<init>\n 0 0 0.0 0.0 0 0 0 0 3 1\n %.8e 0.0 1.0 1\n</init>\n'
                    % cross)
            for cand in events:
                f.write(self.lhe_event(cand, scale) + '\n')
            f.write('</LesHouchesEvents>\n')


def _rambo(M, masses):
    """RAMBO momenta (dict (k, i), k = 1..3 the components, 4 the energy) and
    log weight of a decay at rest; the phase space is exp(weight)
    (2 pi)^(4 - 3n)."""
    import madgraph.various.rambo as rambo
    xm = rambo.FortranList(len(masses))
    for i, m in enumerate(masses):
        xm[i + 1] = m
    return rambo.RAMBO(len(masses), M, xm)


def lo_width(term, library, M, masses, alphas, rng=None, nsample=4000):
    """The LO partial width of the ordinary decay of a bound state of mass M
    (term, a Terminal with its library order), from the helicity-summed |M|^2
    of the density library (smatrixhel, with its average over the initial
    helicities and its identical-particle factor): Gamma = |p| <|M|^2> /
    (8 pi M^2) for two products, the phase-space average with RAMBO for more."""
    import random
    rng = rng or random.Random(1)
    order = list(term.order)
    ms = [masses.get(abs(q), 0.) for q in order[1:]]
    n = len(ms)

    def me(moms):
        P = numpy.asfortranarray(numpy.array([[M, 0., 0., 0.]] + moms, dtype=float).T)
        return float(library.smatrixhel(order, 0, P, alphas, M * M, -1))
    if n == 2:
        q = math.sqrt(max((M * M - (ms[0] + ms[1]) ** 2) * (M * M - (ms[0] - ms[1]) ** 2), 0.)) / (2 * M)
        vals = []
        for _ in range(min(nsample, 200)):
            p1, p2 = two_body(M, ms[0], ms[1], random_direction(rng))
            vals.append(me([list(p1), list(p2)]))
        return q * sum(vals) / len(vals) / (8 * math.pi * M * M)
    total = 0.
    for _ in range(nsample):
        P, lwt = _rambo(M, ms)
        moms = [[P[(4, i)], P[(1, i)], P[(2, i)], P[(3, i)]] for i in range(1, n + 1)]
        total += me(moms) * math.exp(lwt)
    return total / nsample * (2 * math.pi) ** (4 - 3 * n) / (2 * M)


# ------------------------------------------------------------- the library --
def set_onium_masses(onia_card_text, masses):
    """The onia card with the entries of its Block onium_mass set to masses
    ({pdg: mass}); the other blocks (the LDMEs) are kept."""
    import re
    out, block = [], None
    for line in onia_card_text.splitlines():
        head = re.match(r'\s*block\s+(\w+)', line, re.I)
        if head:
            block = head.group(1).lower()
        elif block == 'onium_mass':
            fields = line.split('#')[0].split()
            if len(fields) >= 2 and fields[0].lstrip('-').isdigit() and abs(int(fields[0])) in masses:
                comment = line.split('#', 1)[1] if '#' in line else ''
                line = '    %s  %.10e   #%s' % (fields[0], masses[abs(int(fields[0]))], comment)
        out.append(line)
    return '\n'.join(out) + '\n'


def build_library(mg5cmd, terminals, path, param_card_text, masses, menum=4, nb_core=1):
    """output standalone_fortran --density of the terminal decays, compiled
    as all_matrix<menum>py and loaded; sets the order of every Terminal.
    The onia card is the one of the output (the decay LDMEs of the decaying
    states), at the masses of the chains. Returns the module (None without
    terminal decays)."""
    import os
    import glob
    import importlib.util
    import madgraph.various.misc as misc
    procs = []
    for t in terminals:
        if t.process not in procs:
            procs.append(t.process)
    if not procs:
        return None
    # ungrouped, as MadSpin's density outputs; the options are put back for
    # the madevent decays that may follow
    saved = dict((key, mg5cmd.options.get(key)) for key in ('group_subprocesses', 'apply_flavor_grouping'))
    mg5cmd.exec_cmd('set group_subprocesses False', printcmd=False, precmd=True)
    mg5cmd.exec_cmd('set apply_flavor_grouping False', printcmd=False, precmd=True)
    try:
        mg5cmd.exec_cmd('generate %s' % procs[0], printcmd=False, precmd=True)
        for proc in procs[1:]:
            mg5cmd.exec_cmd('add process %s' % proc, printcmd=False, precmd=True)
        mg5cmd.exec_cmd('output standalone_fortran %s --prefix=int --density=1 -f' % path,
                        printcmd=False, precmd=True)
    finally:
        for key, value in saved.items():
            if value is not None:
                mg5cmd.exec_cmd('set %s %s' % (key, value), printcmd=False, precmd=True)
    cards = os.path.join(path, 'Cards')
    with open(os.path.join(cards, 'param_card.dat'), 'w') as f:
        f.write(param_card_text)
    onia_card = os.path.join(cards, 'onia_card.dat')
    if os.path.exists(onia_card):
        text = set_onium_masses(open(onia_card).read(), masses)
        with open(onia_card, 'w') as f:
            f.write(text)
    misc.compile(cwd=os.path.join(path, 'Source'), nb_core=nb_core)
    sp = os.path.join(path, 'SubProcesses')
    misc.compile(['MENUM=%d' % menum, 'all_matrix%dpy.so' % menum], cwd=sp, nb_core=nb_core)
    libs = [f for f in glob.glob(os.path.join(sp, 'all_matrix%dpy*.so' % menum)) if os.path.getsize(f)]
    spec = importlib.util.spec_from_file_location('all_matrix%dpy' % menum, libs[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with misc.chdir(sp):
        mod.initialise(os.path.join(cards, 'param_card.dat'))
    pdg_orders, _ = mod.get_pdg_order()
    orders = [tuple(int(x) for x in row if int(x) != 0) for row in pdg_orders]
    for t in terminals:
        for order in orders:
            if order[0] == t.fock.get('pdg_code') and sorted(order[1:]) == sorted(t.products):
                t.order = order
                break
        else:
            raise RuntimeError('MadSpin: %s is not in the library of the radiative chains (%s)'
                               % (t.process, orders))
    return mod
