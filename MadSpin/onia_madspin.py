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
"""Bound states (onia) in MadSpin.

A quarkonium or leptonium is not a particle of the model: its Fock states
are bound states of the generator (mg5cmd._fockstates), and its name, e.g.
'Jpsi(1|3S11)', has the parentheses that MadSpin's decay-chain syntax uses for
grouping. MadSpin's bookkeeping (labels, pdg codes, spin, colour, masses) goes
through the model, so each Fock state gets

  * an alias without parentheses, 'Jpsi_1_3S11', which the generator knows as a
    bound-state label (alias -> that Fock state), so that 'generate
    Jpsi_1_3S11 > mu+ mu-' is the decay of the bound state;
  * a pseudo-particle of that name in a copy of the model used for MadSpin's
    bookkeeping only (pdg code, spin 2J+1, colour, charge), never by the
    generator, which keeps its own model.

A decay line names the hadron ('decay Jpsi > mu+ mu-') or a Fock state
('decay Jpsi(1|3S11) > mu+ mu-'); the decay is that of the colour-singlet Fock
state of the hadron, at the onium_mass of the production run.
"""
from __future__ import absolute_import
import copy
import re

import madgraph.core.base_objects as base_objects

FOCK_RE = re.compile(r'([A-Za-z][\w*+\-~]*)\((\d+)\|(\w+)\)')


def alias(name):
    """'Jpsi(1|3S11)' -> 'Jpsi_1_3S11' (a name without parentheses)."""
    return name.replace('(', '_').replace('|', '_').replace(')', '')


# ------------------------------------------------- spin of colour octets --
def clebsch_gordan(j1, m1, j2, m2, J, M):
    """<j1 m1; j2 m2 | J M> for integer angular momenta (Condon-Shortley,
    the convention of CLEBSCH_GORDAN in aloha_functions_dual.f, which builds
    the J_z states of the P-wave bound states)."""
    from math import factorial, sqrt
    if m1 + m2 != M or abs(m1) > j1 or abs(m2) > j2 or abs(M) > J:
        return 0.0
    if J < abs(j1 - j2) or J > j1 + j2:
        return 0.0
    pref = sqrt((2 * J + 1) * factorial(J + j1 - j2) * factorial(J - j1 + j2) *
                factorial(j1 + j2 - J) / factorial(j1 + j2 + J + 1))
    pref *= sqrt(factorial(J + M) * factorial(J - M) * factorial(j1 - m1) *
                 factorial(j1 + m1) * factorial(j2 - m2) * factorial(j2 + m2))
    total = 0.0
    for k in range(0, j1 + j2 + J + 1):
        args = (k, j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k, J - j2 + m1 + k, J - j1 - m2 + k)
        if min(args) < 0:
            continue
        den = 1
        for a in args:
            den *= factorial(a)
        total += (-1) ** k / den
    return pref * total


def _linear_transfer(fock, hadron, rho, fock_hels, hadron_hels):
    """spin_transfer as a linear map (the M1 case as Tr(rho)/(2J_H+1) times
    the identity, the others unchanged), without the final rescaling to the
    trace of rho: apply_transfer builds the map from basis matrices."""
    import numpy
    S = (fock.get('spin') - 1) // 2
    SH = (hadron.get('spin') - 1) // 2
    if S != SH:
        n = len(hadron_hels)
        return numpy.eye(n, dtype=complex) * numpy.trace(rho) / n
    return spin_transfer(fock, hadron, rho, fock_hels, hadron_hels, renormalise=False)


def spin_transfer(fock, hadron, rho, fock_hels, hadron_hels, rho_uncoupled=None,
                  renormalise=True):
    """The spin-density matrix of the hadron produced by the soft transition
    of a colour-octet Fock state, at LO in v by heavy-quark spin symmetry:
    the transition is E1 (the heavy-quark spin S kept) or M1 (S changed).

      * S different from S_H (1S0[8] > J/psi, 3S1[8] > eta_c): the spin is
        isotropic, rho_H = Tr(rho) 1/(2J_H+1);
      * L = L_H = 0 (3S1[8] > J/psi): rho_H = rho (J_z = S_z);
      * L = 0, L_H = 1 (3S1[8] > chi_J, 1S0[8] > h_c): the new orbital state is
        isotropic, rho_H = sum_Lz CG CG rho / 3 (the CG of L_H = 1 with S);
      * L = 1, L_H = 0 (3PJ[8] > J/psi, 1P1[8] > eta_c): rho_H is the partial
        trace over L_z of the density in L_z x S_z. rho_uncoupled is that
        density (from the uncoupled P-wave matrix element, which has the
        interference between different J); without it the J-resolved density
        is expanded with its CG coefficients (no interference between J).

    rho is in the order of fock_hels (the J_z of the Fock state, or for
    rho_uncoupled the (L_z, S_z) pairs), the result in the order of
    hadron_hels; its trace is that of rho, the production |M|^2 of the
    event (the rate is the production's)."""
    import numpy
    S, L, J = (fock.get('spin') - 1) // 2, fock.get('orbital'), fock.get('J')
    SH, LH, JH = (hadron.get('spin') - 1) // 2, hadron.get('orbital'), hadron.get('J')
    nH = len(hadron_hels)
    trace = numpy.trace(rho).real
    if fock is hadron or (S, L, J) == (SH, LH, JH) and L == 0:
        out = numpy.array(rho, dtype=complex)
    elif S != SH:
        out = numpy.eye(nH, dtype=complex) / nH * trace
    elif L == 0 and LH == 0:
        out = numpy.array(rho, dtype=complex)
    elif L == 0 and LH == 1:
        out = numpy.zeros((nH, nH), dtype=complex)
        for a, M in enumerate(hadron_hels):
            for b, Mp in enumerate(hadron_hels):
                v = 0j
                for i, Sz in enumerate(fock_hels):
                    for k, Szp in enumerate(fock_hels):
                        for Lz in (-1, 0, 1):
                            v += (clebsch_gordan(1, Lz, S, Sz, JH, M) *
                                  clebsch_gordan(1, Lz, S, Szp, JH, Mp) * rho[i, k])
                out[a, b] = v / 3
    elif L == 1 and LH == 0:
        if rho_uncoupled is None:
            # (L_z, S_z) density of this J alone
            pairs = [(Lz, Sz) for Lz in (-1, 0, 1) for Sz in range(-S, S + 1)]
            unc = numpy.zeros((len(pairs), len(pairs)), dtype=complex)
            for a, (Lz, Sz) in enumerate(pairs):
                for b, (Lzp, Szp) in enumerate(pairs):
                    v = 0j
                    for i, M in enumerate(fock_hels):
                        for k, Mp in enumerate(fock_hels):
                            v += (clebsch_gordan(1, Lz, S, Sz, J, M) *
                                  clebsch_gordan(1, Lzp, S, Szp, J, Mp) * rho[i, k])
                    unc[a, b] = v
        else:
            pairs, unc = rho_uncoupled
        out = numpy.zeros((nH, nH), dtype=complex)
        for a, Sz in enumerate(hadron_hels):
            for b, Szp in enumerate(hadron_hels):
                out[a, b] = sum(unc[pairs.index((Lz, Sz)), pairs.index((Lz, Szp))]
                                for Lz in (-1, 0, 1)
                                if (Lz, Sz) in pairs and (Lz, Szp) in pairs)
    else:
        raise NotImplementedError('MadSpin: no spin transfer from %s to %s'
                                  % (fock.get('name'), hadron.get('name')))
    if not renormalise:
        return out
    t = numpy.trace(out).real
    return out * (trace / t) if t else out


class OniaRegistry(object):
    """The Fock states of the generator, by name, alias and pdg code."""

    def __init__(self, mg5cmd):
        self.fockstates = list(getattr(mg5cmd, '_fockstates', []) or [])
        self.labels = dict(getattr(mg5cmd, '_boundstates', {}) or {})
        self.by_name = {}
        self.by_pdg = {}
        self.by_alias = {}
        for f in self.fockstates:
            self.by_name[f.get('name')] = f
            self.by_pdg[f.get('pdg_code')] = f
            self.by_alias[alias(f.get('name'))] = f

    def __bool__(self):
        return bool(self.fockstates)
    __nonzero__ = __bool__

    # ------------------------------------------------------------- lookups --
    @staticmethod
    def quantum_numbers(fock):
        """(2S+1, L, J, colour, n) of a Fock state."""
        return (fock.get('spin'), fock.get('orbital'), fock.get('J'),
                fock.get('color'), fock.get('principal'))

    def is_onium(self, pdg):
        return pdg in self.by_pdg

    def hadron_name(self, fock):
        """The hadron a Fock state belongs to: 'Jpsi' for 'Jpsi(1|3S18)', with
        its principal number (the Fock states of 'psi(2|...)' are psi(2S))."""
        m = FOCK_RE.match(fock.get('name'))
        return (m.group(1), int(m.group(2))) if m else (fock.get('name'), None)

    def singlet_of(self, fock):
        """The colour-singlet Fock state of the hadron of a Fock state (the
        physical bound state at LO in v): same name and principal number,
        colour 1. A leptonium is its own."""
        if fock.get('color') == 1:
            return fock
        name, n = self.hadron_name(fock)
        for label, states in self.labels.items():
            if fock.get('name') in states:
                for s in states:
                    g = self.by_name.get(s)
                    if g is not None and g.get('color') == 1 and self.hadron_name(g) == (name, n):
                        return g
        for g in self.fockstates:
            if g.get('color') == 1 and self.hadron_name(g) == (name, n):
                return g
        return None

    def fock_of_name(self, name):
        """A Fock state from a decay-line name: a Fock-state name or alias, or
        a hadron label of the generator ('Jpsi': its colour-singlet state)."""
        if name in self.by_name:
            return self.by_name[name]
        if name in self.by_alias:
            return self.by_alias[name]
        if name in self.labels:
            states = [self.by_name[s] for s in self.labels[name] if s in self.by_name]
            singlets = [s for s in states if s.get('color') == 1]
            if singlets:
                return singlets[0]
        return None

    def octets_of(self, singlet):
        """The colour-octet Fock states of the hadron of a colour singlet."""
        return [f for f in self.fockstates if f.get('color') != 1
                and self.singlet_of(f) is singlet]

    def relabel_octets(self, event, masses=None):
        """In a decayed event, a colour-octet Fock state becomes its hadron
        (the soft gluons of the transition are not generated): the pdg code of
        the hadron, its three-momentum kept and its energy at the mass of the
        hadron (the invariant mass of its decay products). With the onium_mass
        of the octet equal to that of the hadron the event keeps its energy."""
        import math
        for particle in event:
            fock = self.by_pdg.get(int(particle.pid))
            if fock is None or fock.get('color') == 1 or particle.status != 2:
                continue
            singlet = self.singlet_of(fock)
            if singlet is None:
                continue
            daughters = [p for p in event if p.mother1 is particle or p.mother2 is particle]
            if daughters:
                E = sum(p.E for p in daughters)
                px = sum(p.px for p in daughters)
                py = sum(p.py for p in daughters)
                pz = sum(p.pz for p in daughters)
                mass = math.sqrt(max(E * E - px * px - py * py - pz * pz, 0.))
            else:
                mass = (masses or {}).get(abs(singlet.get('pdg_code')), particle.mass)
            particle.pid = singlet.get('pdg_code')
            particle.mass = mass
            particle.E = math.sqrt(particle.px ** 2 + particle.py ** 2 + particle.pz ** 2 + mass * mass)

    # ------------------------------------- production density of an octet --
    def transfer_plan(self, pdgs, position, allow_hel):
        """For a production density whose decaying legs (position) include a
        colour-octet Fock state: the helicities to ask the matrix element for
        (the J_z of each Fock state) and what spin_transfer needs. None when no
        leg needs a transfer."""
        n = len(position)
        combos = [tuple(allow_hel[i * n:(i + 1) * n]) for i in range(len(allow_hel) // n)]
        hadron_hels = [sorted(set(c[k] for c in combos), key=[c[k] for c in combos].index)
                       for k in range(n)]
        legs, need = [], False
        for k, pos in enumerate(position):
            fock = self.by_pdg.get(int(pdgs[pos - 1]))
            if fock is None or fock.get('color') == 1:
                legs.append(None)
                continue
            need = True
            J = fock.get('J')
            legs.append((fock, self.singlet_of(fock), list(range(-J, J + 1))
                         if J else [0]))
        if not need:
            return None
        fock_hels = [leg[2] if leg else hadron_hels[k] for k, leg in enumerate(legs)]
        # a 3P octet of a hadron in an S wave: its density in L_z x S_z (the
        # uncoupled matrix element, helicity codes -4..4) has the interference
        # between its J; spin_transfer takes the partial trace over L_z
        unc_legs = [k for k, leg in enumerate(legs) if leg and self.needs_uncoupled(leg[0])]
        unc_hels = [list(range(-4, 5)) if k in unc_legs else fock_hels[k] for k in range(n)]
        import itertools
        fock_combos = list(itertools.product(*fock_hels))
        fock_allow = [h for c in fock_combos for h in c]
        unc_allow = [h for c in itertools.product(*unc_hels) for h in c] if unc_legs else None
        return dict(n=n, legs=legs, hadron_hels=hadron_hels, fock_hels=fock_hels,
                    fock_combos=fock_combos, hadron_combos=combos, fock_allow=fock_allow,
                    unc_legs=unc_legs, unc_hels=unc_hels, unc_allow=unc_allow)

    def needs_uncoupled(self, fock):
        """A 3P colour octet whose hadron is in an S wave: its spin goes to the
        hadron through the partial trace of its L_z x S_z density."""
        hadron = self.singlet_of(fock)
        return (fock.get('color') != 1 and fock.get('orbital') == 1 and fock.get('spin') == 3
                and hadron is not None and hadron.get('orbital') == 0)

    def production_needs_uncoupled(self, ms, processes):
        """Whether MadSpin needs the uncoupled production (a decayed hadron
        produced as a 3P octet in one of the processes)."""
        decayed = set(abs(f.get('pdg_code')) for f in self.decaying(ms))
        for proc in processes:
            for leg in proc.get('legs'):
                onium = leg.get('onium')
                if not onium:
                    continue
                fock = self.by_pdg.get(onium.get('id'))
                if fock and self.needs_uncoupled(fock) and \
                        abs(self.singlet_of(fock).get('pdg_code')) in decayed:
                    return True
        return False

    @staticmethod
    def unpack(packed, N):
        import numpy
        rho = numpy.zeros((N, N), dtype=complex)
        k = 0
        for i in range(N):
            for j in range(i, N):
                rho[i, j] = packed[k]
                rho[j, i] = numpy.conj(packed[k])
                k += 1
        return rho

    @staticmethod
    def pack(rho):
        N = rho.shape[0]
        return [rho[i, j] for i in range(N) for j in range(i, N)]

    def apply_transfer(self, plan, packed, uncoupled=False, target_trace=None):
        """The production density in the hadron J_z of every colour-octet
        leg (spin_transfer on that leg, the identity on the others), packed as
        the matrix element packs it, in the order of the requested helicities.
        With uncoupled, packed is the density of the uncoupled matrix element
        (plan['unc_hels']); target_trace, the trace to keep (the |M|^2 of the
        event's own matrix element), else that of packed."""
        import numpy
        hels = plan['unc_hels'] if uncoupled else plan['fock_hels']
        nF = [len(h) for h in hels]
        rho = self.unpack(packed, int(numpy.prod(nF)))
        n = plan['n']
        tensor = rho.reshape(nF + nF)
        for k, leg in enumerate(plan['legs']):
            if leg is None:
                continue
            fock, hadron, fhel = leg
            hhel = plan['hadron_hels'][k]
            uncoupled_leg = uncoupled and k in plan['unc_legs']
            if uncoupled_leg:
                fhel = hels[k]
            # the map on leg k, linear in rho: apply it to the basis matrices
            dimF, dimH = len(fhel), len(hhel)
            T = numpy.zeros((dimH, dimH, dimF, dimF), dtype=complex)
            if uncoupled_leg:
                # the codes -4..4 are the (L_z, S_z) pairs; partial trace over L_z
                pairs = [((h + 4) // 3 - 1, (h + 4) % 3 - 1) for h in fhel]
                for x, Sz in enumerate(hhel):
                    for y, Szp in enumerate(hhel):
                        for Lz in (-1, 0, 1):
                            if (Lz, Sz) in pairs and (Lz, Szp) in pairs:
                                T[x, y, pairs.index((Lz, Sz)), pairs.index((Lz, Szp))] = 1.
            else:
              for a in range(dimF):
                for b in range(dimF):
                    e = numpy.zeros((dimF, dimF), dtype=complex)
                    e[a, b] = 1.
                    T[:, :, a, b] = _linear_transfer(fock, hadron, e, fhel, hhel)
            # move leg k axes (k, n + k) to the front, contract, move back
            t = numpy.moveaxis(tensor, (k, n + k), (0, 1))
            t = numpy.tensordot(T, t, axes=([2, 3], [0, 1]))
            tensor = numpy.moveaxis(t, (0, 1), (k, n + k))
            nF[k] = dimH
        N = int(numpy.prod(nF))
        rho_h = tensor.reshape(N, N)
        # the requested order of the hadron combinations is the product order
        t_in = numpy.trace(rho).real if target_trace is None else target_trace
        t_out = numpy.trace(rho_h).real
        if t_out:
            rho_h *= t_in / t_out
        return self.pack(rho_h)

    def antiparticle(self, fock):
        """The charge conjugate Fock state (itself for a neutral onium)."""
        if fock.get('charge') == 0:
            return fock
        return self.by_pdg.get(-fock.get('pdg_code'), fock)

    # ---------------------------------------------- MadSpin's decay lines --
    def rewrite_decay_line(self, line):
        """The parent of a decay line ('Jpsi > mu+ mu-', 'Jpsi(1|3S11) > ...')
        replaced by the alias of the colour-singlet Fock state of its hadron,
        which the generator decays; None if the parent is no bound state."""
        parent, sep, rest = line.strip().partition('>')
        parent = parent.strip()
        fock = self.fock_of_name(parent)
        if fock is None:
            return None
        singlet = self.singlet_of(fock)
        if singlet is None:
            raise ValueError('MadSpin: no colour-singlet Fock state for %s' % parent)
        return '%s > %s' % (alias(singlet.get('name')), rest.strip())

    def register_aliases(self, mg5cmd):
        """Make every alias a bound-state label of the generator."""
        for f in self.fockstates:
            mg5cmd._boundstates.setdefault(alias(f.get('name')), [f.get('name')])

    # ---------------------------------------------------- the model copy --
    def pseudo_particle(self, fock):
        anti = self.antiparticle(fock)
        neutral = anti is fock
        particle = base_objects.Particle({
            # a colour octet decays as its hadron (spin_transfer): the spin and
            # colour of the hadron
            'spin': 2 * (self.singlet_of(fock) or fock).get('J') + 1,
            'color': (self.singlet_of(fock) or fock).get('color'),
            'charge': float(fock.get('charge')),
            'mass': 'ZERO',
            'width': 'ZERO',
            'pdg_code': abs(fock.get('pdg_code')),
            'line': 'dashed',
            'propagating': True,
            'is_part': True,
            'self_antipart': neutral,
        })
        # set directly: a hadron name such as 'B*c+' is no valid particle name
        # for Particle.set, but MadSpin only looks it up
        dict.__setitem__(particle, 'name', alias(fock.get('name')))
        dict.__setitem__(particle, 'antiname', alias(anti.get('name')))
        return particle

    def bookkeeping_model(self, model):
        """A shallow copy of the model with a pseudo-particle per Fock state
        (one per charge-conjugate pair, with the positive pdg code), for
        MadSpin's bookkeeping only."""
        view = copy.copy(model)
        particles = base_objects.ParticleList(list(model.get('particles')))
        seen = set()
        for f in self.fockstates:
            pdg = abs(f.get('pdg_code'))
            if pdg in seen:
                continue
            seen.add(pdg)
            # the positive code carries the name (Bc+(1|1S01) for 541)
            fock = self.by_pdg.get(pdg, f)
            particles.append(self.pseudo_particle(fock))
        dict.__setitem__(view, 'particles', particles)
        dict.__setitem__(view, 'particle_dict', {})
        for key in ('name2pdg',):
            if key in view:
                dict.__delitem__(view, key)
        if hasattr(view, 'name2part'):
            del view.name2part
        view._onia_registry = self
        return view

    # --------------------------------------------------------- launching --
    def decaying(self, ms):
        """The Fock states MadSpin decays (the parents of its decay lines)."""
        return [self.by_alias[name] for name in ms.list_branches if name in self.by_alias]

    def check_launch(self, ms):
        """Before MadSpin runs with bound states to decay: their widths, and
        the backends that support them."""
        decaying = self.decaying(ms)
        if not decaying:
            return
        import logging
        logger = logging.getLogger('decay.stdout')
        if not hasattr(ms.banner, 'param_card'):
            ms.banner.charge_card('slha')
        for fock in decaying:
            pdg = abs(fock.get('pdg_code'))
            try:
                ms.banner.param_card['decay'].get((pdg,))
            except Exception:
                raise ms.InvalidCmd(
                    "MadSpin: %s (pdg %d) is a bound state, with no width in the model. Add its "
                    "total width to the param_card of the events (a DECAY block, e.g. 'DECAY %d "
                    "9.29e-05' for the J/psi), with optional BR lines; without BR lines MadSpin "
                    "computes the LO partial width of each decay." % (fock.get('name'), pdg, pdg))
        # The colour-octet Fock states of a decaying hadron decay as the hadron
        # (its colour-singlet decay, its width), with the spin density of the
        # octet carried over by heavy-quark spin symmetry (spin_transfer).
        import madgraph.models.check_param_card as check_param_card
        decay_block = ms.banner.param_card['decay']
        for singlet in decaying:
            width = decay_block.get((abs(singlet.get('pdg_code')),)).value
            for octet in self.octets_of(singlet):
                name = alias(octet.get('name'))
                if name not in ms.list_branches:
                    ms.list_branches[name] = list(ms.list_branches[alias(singlet.get('name'))])
                pdg = abs(octet.get('pdg_code'))
                try:
                    decay_block.get((pdg,))
                except KeyError:
                    decay_block.append(check_param_card.Parameter(
                        block='decay', lhacode=[pdg], value=width,
                        comment='width of %s (colour octet)' % singlet.get('name')))
            for name, value, hqss in self.hqss_ldme_mismatch(
                    dict.get(ms.banner, 'mgoniacard', ''), singlet):
                logger.warning(
                    'MadSpin: the LDME of %s in the onia card of the events is %g, not the %g '
                    'of heavy-quark spin symmetry, (2J+1) <O(3P0[8])>. The spin of the %s comes '
                    'from the 3P[8] density summed over J with the HQSS LDMEs, so its decays '
                    'are those of HQSS, while the rate of each J is that of the events.'
                    % (name, value, hqss, singlet.get('name')))
            ldme = self.onia_block(dict.get(ms.banner, 'mgoniacard', ''), 'ldme')
            js = sorted(f.get('J') for f in self.octets_of(singlet)
                        if self.needs_uncoupled(f) and abs(f.get('pdg_code')) in ldme)
            if js and js != [0, 1, 2]:
                logger.warning(
                    'MadSpin: the events have the 3PJ[8] states of the %s with J = %s only. Its '
                    'spin comes from the 3P[8] density summed over J = 0, 1, 2 (the interference '
                    'between J included, as heavy-quark spin symmetry requires), which describes '
                    'the sum of the three J; generate all three (e.g. Jpsi(1|3PJ8)).'
                    % (singlet.get('name'), ', '.join(str(j) for j in js)))
        # the density matrices of bound states are written in Fortran
        # (madmatrix has no bound states), and their decays generated by
        # madevent (mg7 neither)
        try:
            if ms.options['density_backend'] != 'fortran':
                logger.info('MadSpin: bound states, the density matrices come from the Fortran '
                            'matrix elements')
                ms.options['density_backend'] = 'fortran'
        except KeyError:
            pass
        try:
            if ms.options['decay_generator'] != 'madevent':
                logger.info('MadSpin: bound states, their decays are generated with madevent')
                ms.options['decay_generator'] = 'madevent'
        except KeyError:
            pass

    def card_branching_ratio(self, ms, pdg):
        """The branching ratio of the decays MadSpin generates for a bound
        state, from the BR lines of its DECAY block in the param_card of the
        events (measured branching ratios, where the LO partial widths of NRQCD
        can be far off), summed over its decay lines; None when a decay line
        has no BR line (MadSpin then uses the LO partial width)."""
        fock = self.by_pdg.get(pdg) or self.by_pdg.get(-pdg)
        if fock is None:
            return None
        singlet = self.singlet_of(fock) or fock
        branches = ms.list_branches.get(alias(singlet.get('name')), [])
        if not hasattr(ms.banner, 'param_card'):
            ms.banner.charge_card('slha')
        table = ms.banner.param_card['decay'].decay_table.get(abs(singlet.get('pdg_code')))
        if not table or not branches:
            return None
        name2pdg = ms._bookkeeping_model().get('name2pdg')
        total = 0.0
        for proc in branches:
            final = proc.split('>', 1)[1].split(',')[0]
            names = [t for t in final.split() if '=' not in t and t not in ('/', '$', '$$')]
            if '/' in final or '$' in final:
                names = [t for t in final.split('/')[0].split('$')[0].split() if '=' not in t]
            try:
                ids = [name2pdg[n] for n in names]
            except KeyError:
                return None
            entry = None
            for order in (sorted(ids), ids):
                try:
                    entry = table.get(tuple([len(order)] + list(order)))
                    break
                except KeyError:
                    continue
            if entry is None:
                return None
            total += entry.value
        return total

    # ------------------------------------------------------------- masses --
    def add_masses_to_banner(self, banner):
        """MadSpin reads a mass from the param_card of the banner: a bound
        state has its onium_mass in the onia card (Block onium_mass), which is
        added to the MASS block of the banner's param_card (in memory)."""
        import madgraph.models.check_param_card as check_param_card
        if not hasattr(banner, 'param_card'):
            banner.charge_card('slha')
        masses = self.onia_masses(dict.get(banner, 'mgoniacard', ''))
        block = banner.param_card['mass']
        for pdg, mass in masses.items():
            for code in set([pdg, abs(pdg)]):
                try:
                    block.get((code,))
                    continue
                except Exception:
                    pass
                block.append(check_param_card.Parameter(block='mass', lhacode=[code], value=mass,
                                                        comment='onium_mass of the onia card'))
        return masses

    @staticmethod
    def write_onia_card(banner, cards_dir):
        """The onia card of the events (LDMEs, decay LDMEs, masses) in a Cards
        directory of MadSpin, next to its param_card: every matrix element and
        every decay is then evaluated with the inputs of the production run."""
        import os
        text = dict.get(banner, 'mgoniacard') if banner is not None else None
        if text and os.path.isdir(cards_dir):
            with open(os.path.join(cards_dir, 'onia_card.dat'), 'w') as fsock:
                fsock.write(text)

    @staticmethod
    def onia_block(onia_card_text, name):
        """{pdg: value} of a block of an onia card ('onium_mass', 'ldme')."""
        out = {}
        block = re.search(r'(?ims)^block\s+%s\b(.*?)(?=^block|\Z)' % name, onia_card_text or '')
        if block:
            for line in block.group(1).splitlines():
                line = line.split('#')[0].split()
                if len(line) >= 2:
                    out[int(line[0])] = float(line[1].replace('d', 'e').replace('D', 'E'))
        return out

    @classmethod
    def onia_masses(cls, onia_card_text):
        """{pdg: onium_mass} of an onia card (Block onium_mass)."""
        return cls.onia_block(onia_card_text, 'onium_mass')

    def hqss_ldme_mismatch(self, onia_card_text, singlet):
        """The 3PJ[8] Fock states of a hadron in an S wave share one
        (L_z, S_z) density, of the uncoupled matrix element, normalised with
        <O(3P0[8])>: their LDMEs must follow heavy-quark spin symmetry,
        <O(3PJ[8])> = (2J+1) <O(3P0[8])>. The [(name, LDME, HQSS value)] of
        those of the onia card which do not (at 1e-6)."""
        ldme = self.onia_block(onia_card_text, 'ldme')
        states = [f for f in self.octets_of(singlet) if self.needs_uncoupled(f)
                  and abs(f.get('pdg_code')) in ldme]
        zero = [f for f in states if f.get('J') == 0]
        if not zero:
            return []
        o0 = ldme[abs(zero[0].get('pdg_code'))]
        out = []
        for f in states:
            value = ldme[abs(f.get('pdg_code'))]
            hqss = (2 * f.get('J') + 1) * o0
            if abs(value - hqss) > 1e-6 * max(abs(hqss), abs(value)):
                out.append((f.get('name'), value, hqss))
        return out
