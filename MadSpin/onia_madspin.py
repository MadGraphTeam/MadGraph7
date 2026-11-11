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
        # radiative transitions (onia_radiative): decay-line text -> Chain
        self.radiative = {}
        self.radiative_evaluator = None
        # the widths and branching fractions of the onia (PDG 2026,
        # madgraph/models/onia_decays.py, or the model's own
        # all_boundstate_decays), for radiative transitions that the
        # param_card of the events does not give
        from madgraph.models.onia_decays import boundstate_decays
        self.default_decays = boundstate_decays(getattr(mg5cmd, '_curr_model', None))

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
            if any(p.pid == 21 and (p.mother1 is particle or p.mother2 is particle) for p in event):
                # decayed to its hadron and a soft gluon (onia_radiative): it
                # stays in the record, the gluon carrying its colour
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
        3P colour octet of an S-wave hadron: it decays (to its hadron and a
        soft gluon, onia_radiative) in the uncoupled (L_z, S_z) basis, whose
        density (helicity codes -4..4) has the interference between its J and
        comes from the uncoupled matrix element, normalised to the |M|^2 of
        the event's own J (the J_z of the Fock state). None when no leg needs
        it: every other colour octet decays in its own J_z."""
        n = len(position)
        combos = [tuple(allow_hel[i * n:(i + 1) * n]) for i in range(len(allow_hel) // n)]
        hadron_hels = [sorted(set(c[k] for c in combos), key=[c[k] for c in combos].index)
                       for k in range(n)]
        legs, need = [], False
        for k, pos in enumerate(position):
            fock = self.by_pdg.get(int(pdgs[pos - 1]))
            if fock is None or fock.get('color') == 1 or not self.needs_uncoupled(fock):
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
                # the codes -4..4 of the (L_z, S_z) pairs, kept: the decay
                # (onia_radiative) is written in the same basis
                for x, hx in enumerate(hhel):
                    for y, hy in enumerate(hhel):
                        if hx in fhel and hy in fhel:
                            T[x, y, fhel.index(hx), fhel.index(hy)] = 1.
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
        # a colour octet decays to its hadron and a soft gluon (onia_radiative)
        # in its own J_z, or a 3P octet of an S-wave hadron in the nine (L_z,
        # S_z) pairs of the uncoupled basis
        uncoupled = self.needs_uncoupled(fock)
        particle = base_objects.Particle({
            'spin': 2 * fock.get('J') + 1,
            'color': fock.get('color'),
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
        if uncoupled:
            # nine helicity codes h = 3(L_z+1)+(S_z+1)-4 (interface_madspin's
            # hel_dict), not a spin Particle.set accepts
            dict.__setitem__(particle, 'spin', 9)
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
                if self.radiative_default_width(ms, fock, logger):
                    continue
                raise ms.InvalidCmd(
                    "MadSpin: %s (pdg %d) is a bound state, with no width in the model. Add its "
                    "total width to the param_card of the events (a DECAY block, e.g. 'DECAY %d "
                    "9.26e-05' for the J/psi), with optional BR lines; without BR lines MadSpin "
                    "computes the LO partial width of each decay." % (fock.get('name'), pdg, pdg))
        # The colour-octet Fock states of a decaying hadron turn into it by
        # emitting a soft gluon (chromo-E1, chromo-M1, or with its spins kept),
        # which takes their colour, then decay as the hadron
        # (onia_radiative): the spin of the octet is passed on as heavy-quark
        # spin symmetry requires, and the event keeps its colour flow.
        import madgraph.models.check_param_card as check_param_card
        decay_block = ms.banner.param_card['decay']
        # the octets of the production run (Block onium_mass of its onia card)
        produced = set(abs(int(p)) for p in self.onia_masses(dict.get(ms.banner, 'mgoniacard', '') or ''))
        for singlet in decaying:
            width = decay_block.get((abs(singlet.get('pdg_code')),)).value
            for octet in self.octets_of(singlet):
                if produced and abs(octet.get('pdg_code')) not in produced:
                    continue
                self.octet_chains(ms, singlet, octet)
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
        # radiative transitions: their branching ratios, masses and the density
        # library of the decays that end their chains
        self.check_radiative(ms, decaying)
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

    # ----------------------------------------------- colour octets --
    def octet_chains(self, ms, singlet, octet):
        """The decay lines of a colour-octet Fock state: octet > hadron g (the
        soft gluon, onia_radiative), then each decay line of the hadron (an
        ordinary decay, or a radiative chain)."""
        import MadSpin.onia_radiative as onia_radiative
        kind = onia_radiative.transition_type(octet, singlet, onia_radiative.GLUON)
        if kind is None:
            raise ms.InvalidCmd(
                "MadSpin: %s cannot turn into %s by one soft gluon (neither chromo-E1, chromo-M1, "
                "nor with its L and S kept)" % (octet.get('name'), singlet.get('name')))
        uncoupled = self.needs_uncoupled(octet)
        name2pdg = ms._bookkeeping_model().get('name2pdg')
        texts = []
        for branch in ms.list_branches.get(alias(singlet.get('name')), []):
            if branch in self.radiative:
                sub = self.radiative[branch]
            else:
                process = ms._split_group_tag(branch)[0] if hasattr(ms, '_split_group_tag') else branch
                rhs = process.split('>', 1)[1]
                names = rhs.split()
                bad = [n for n in names if n not in name2pdg]
                if bad or ',' in rhs:
                    raise ms.InvalidCmd(
                        "MadSpin: %s is produced as a colour octet, which decays through its hadron "
                        "and a soft gluon; the decay of the hadron then takes particles of the model "
                        "only, one final state per decay line (not '%s')" % (singlet.get('name'), branch))
                sub = onia_radiative.Terminal(singlet, process.strip(), [name2pdg[n] for n in names])
            text = '%s > %s g, %s' % (alias(octet.get('name')), alias(singlet.get('name')), branch)
            self.radiative[text] = onia_radiative.Chain(octet, singlet, kind, sub, text,
                                                        boson=onia_radiative.GLUON, uncoupled=uncoupled)
            texts.append(text)
        ms.list_branches[alias(octet.get('name'))] = texts

    @staticmethod
    def alpha_s(param_card, scale):
        """alpha_s at scale (one loop, 4 flavours) from aS(m_Z) of the param_card."""
        import math
        try:
            a0 = param_card['sminputs'].get((3,)).value
        except Exception:
            a0 = 0.118
        b0 = (33 - 2 * 4) / (12 * math.pi)
        return a0 / (1 + a0 * b0 * math.log(scale ** 2 / 91.1876 ** 2))

    # ------------------------------------------- radiative transitions --
    # the total width given to a parent whose width is not measured (chi_bJ,
    # h_b): for a bound state decaying by radiative transitions only, its
    # width only normalises the branching ratio, which is that of the BR
    # lines (or of the model), and cancels
    RADIATIVE_PLACEHOLDER_WIDTH = 1e-4

    def radiative_default_width(self, ms, fock, logger):
        """A bound state that decays by radiative transitions only and has no
        DECAY block in the param_card of the events: its DECAY block is added
        (in memory) with the total width of the model (PDG), or a placeholder
        where it is not measured; False if it has other decays, or needs a
        real width (onia_breit_wigner)."""
        import madgraph.models.check_param_card as check_param_card
        name = alias((self.singlet_of(fock) or fock).get('name'))
        branches = ms.list_branches.get(name) or ms.list_branches.get(alias(fock.get('name')), [])
        if not branches or not all(b in self.radiative for b in branches):
            return False
        pdg = abs((self.singlet_of(fock) or fock).get('pdg_code'))
        width = (self.default_decays.get(pdg) or (None, {}))[0]
        if width is None:
            if ms._onia_breit_wigner():
                return False
            width = self.RADIATIVE_PLACEHOLDER_WIDTH
            logger.info('MadSpin: %s has no DECAY block and no measured width: it decays by radiative '
                        'transitions only, whose branching ratios are the BR lines (or the PDG 2026 '
                        'values), so its width (set to %g GeV) only normalises them.'
                        % (fock.get('name'), width))
        else:
            logger.info('MadSpin: %s has no DECAY block: its total width is the PDG 2026 value, %g GeV'
                        % (fock.get('name'), width))
        ms.banner.param_card['decay'].append(check_param_card.Parameter(
            block='decay', lhacode=[abs(fock.get('pdg_code'))], value=width,
            comment='width of %s (model default)' % fock.get('name')))
        return True

    def parse_radiative(self, ms, line):
        """A decay line whose first step is a radiative transition between
        bound states ('chic1 > Jpsi a, Jpsi > mu+ mu-', onia_radiative): its
        Chain is stored and the text MadSpin keeps for it returned (the parent
        under the alias of its colour singlet); None for any other line."""
        import MadSpin.onia_radiative as onia_radiative
        try:
            chain = onia_radiative.parse_chain(self, line, ms._bookkeeping_model().get('name2pdg'))
        except ValueError as error:
            raise ms.InvalidCmd(str(error))
        if chain is None:
            return None
        parts = []
        c = chain
        while isinstance(c, onia_radiative.Chain):
            parts.append('%s > %s a' % (alias(c.parent.get('name')), alias(c.daughter.get('name'))))
            c = c.sub
        if isinstance(c, onia_radiative.Terminal):
            parts.append(c.process)
        text = ', '.join(parts)
        self.radiative[text] = chain
        return alias(chain.parent.get('name')), text

    def radiative_branches(self, ms, pdg):
        """[(index, text, Chain)] of the decay lines of the bound state pdg
        (a Fock state or its antiparticle) that are radiative chains."""
        fock = self.by_pdg.get(pdg) or self.by_pdg.get(-pdg)
        if fock is None:
            return []
        out = []
        for name in (alias(fock.get('name')), alias((self.singlet_of(fock) or fock).get('name'))):
            for i, text in enumerate(ms.list_branches.get(name, [])):
                if text in self.radiative:
                    out.append((i, text, self.radiative[text]))
            if out:
                break
        return out

    def radiative_chain_for_event(self, chain, pdg, ms):
        """The chain as the decay events of the bound state pdg carry it: the
        parent's pdg code is the event's (a colour octet decays as its hadron);
        for an antiparticle every particle of the chain is conjugated."""
        if pdg == chain.parent_pdg:
            return chain
        fock = self.by_pdg.get(abs(pdg))
        conj = None
        if pdg < 0 or (fock is not None and fock.get('charge') != 0 and pdg != abs(pdg)):
            model = ms._bookkeeping_model()

            def conj(q):
                if q in self.by_pdg:
                    anti = self.antiparticle(self.by_pdg[q])
                    return anti.get('pdg_code') if anti is not self.by_pdg[q] else q
                part = model.get_particle(q)
                return q if part is None or part.get('self_antipart') else -q
        return chain.with_parent_pdg(pdg, conj)

    def check_radiative(self, ms, decaying):
        """The checks and the density library of the radiative chains, at
        launch (check_launch)."""
        import os
        import logging
        import MadSpin.onia_radiative as onia_radiative
        logger = logging.getLogger('decay.stdout')
        names = [n for n, branches in ms.list_branches.items()
                 if any(b in self.radiative for b in branches)]
        if not names:
            return
        if ms.options['spinmode'] not in ('madspin', 'full', 'onshell', 'PA'):
            raise ms.InvalidCmd(
                "MadSpin: radiative transitions of bound states and the decays of colour octets "
                "(%s) are decays with spin correlations: use spinmode=madspin (or onshell, PA), "
                "not %s." % (', '.join(names), ms.options['spinmode']))
        for name in names:
            if not all(b in self.radiative for b in ms.list_branches[name]):
                raise ms.InvalidCmd(
                    "MadSpin: %s has radiative and other decay lines; MadSpin decays a bound state "
                    "either by radiative transitions only or by other decays only." % name)
        decay_block = ms.banner.param_card['decay']
        # the masses of the bound states: the onium_mass of the production run
        # (in the MASS block of the banner, add_masses_to_banner), or a MASS
        # line of the param_card; a state that the production does not have
        # (the daughter of a transition) takes the mass of the model
        masses, from_model = {}, []
        for f in self.fockstates:
            pdg = abs(f.get('pdg_code'))
            try:
                masses[pdg] = ms.banner.get('param_card', 'mass', pdg).value
            except Exception:
                if f.get('mass') is not None:
                    masses[pdg] = float(f.get('mass'))
                    from_model.append(f)
        used = set()
        for chain in self.radiative.values():
            for step in chain.steps():
                used.update([abs(step.parent.get('pdg_code')), abs(step.daughter.get('pdg_code'))])
        shown = ['%s %s GeV' % (f.get('name'), masses[abs(f.get('pdg_code'))]) for f in from_model
                 if abs(f.get('pdg_code')) in used]
        if shown:
            logger.info('MadSpin: radiative chains with the masses of the model for %s (a MASS line '
                        'of the param_card of the events sets another)', ', '.join(shown))
        for text, chain in self.radiative.items():
            for step in chain.steps():
                mi = masses.get(abs(step.parent.get('pdg_code')))
                mf = masses.get(abs(step.daughter.get('pdg_code')))
                if mi is None or mf is None or mi <= mf:
                    if step.boson == onia_radiative.GLUON:
                        raise ms.InvalidCmd(
                            "MadSpin: the colour octet %s (%s GeV) turns into %s (%s GeV) by emitting a "
                            "soft gluon, which needs it heavier: set its onium_mass above that of the "
                            "hadron in the onia card of the production run (sm_onia: 0.2 GeV above, "
                            "as Onia:massSplit of Pythia8)." % (
                                step.parent.get('name'), mi, step.daughter.get('name'), mf))
                    raise ms.InvalidCmd(
                        "MadSpin: %s > %s a needs the mass of %s (%s GeV) above that of %s (%s GeV): "
                        "give the bound states their physical masses in Block onium_mass of the onia "
                        "card of the production run." % (
                            step.parent.get('name'), step.daughter.get('name'), step.parent.get('name'),
                            mi, step.daughter.get('name'), mf))
        clash = onia_radiative.ambiguous(list(self.radiative.values()), masses)
        if clash:
            a, b = clash[0]
            raise ms.InvalidCmd(
                "MadSpin: the radiative decay lines '%s' and '%s' have the same final state and "
                "intermediate states of the same mass: their decays cannot be told apart. Give the "
                "bound states their physical masses (Block onium_mass of the onia card, or MASS "
                "lines of the param_card)." % (a.text, b.text))
        terminals = [c.terminal() for c in self.radiative.values()
                     if isinstance(c.terminal(), onia_radiative.Terminal)]
        path = os.path.join(os.path.realpath(ms.options['ms_dir'] or ms.options['curr_dir']),
                            'madspin_radiative')
        library = onia_radiative.build_library(
            ms.mg5cmd, terminals, path, ms.banner['slha'], masses,
            nb_core=ms.mg5cmd.options.get('nb_core') or 1)
        self.radiative_evaluator = onia_radiative.Evaluator(library, masses)
        self.radiative_masses = masses
        # the ordinary decay that ends the chain of a colour octet: without a
        # BR line, its LO partial width over the total width, as MadSpin does
        # for the decays of the colour singlet
        self.radiative_lo_widths = {}
        for chain in self.radiative.values():
            term = chain.terminal()
            if chain.boson != onia_radiative.GLUON or not isinstance(term, onia_radiative.Terminal) \
                    or term.process in self.radiative_lo_widths:
                continue
            table = decay_block.decay_table.get(abs(term.fock.get('pdg_code')))
            if table:
                try:
                    table.get(tuple([len(term.products)] + sorted(term.products)))
                    continue
                except KeyError:
                    pass
            products = dict((abs(q), ms.banner.get('param_card', 'mass', abs(q)).value)
                            for q in term.products if abs(q) not in masses and abs(q) not in (21, 22))
            M = masses[abs(term.fock.get('pdg_code'))]
            width = onia_radiative.lo_width(term, library, M, dict(list(masses.items()) + list(products.items())),
                                            self.alpha_s(ms.banner.param_card, M))
            self.radiative_lo_widths[term.process] = width
            logger.info('MadSpin: %s, no BR line: its LO partial width, %.4g GeV' % (term.process, width))
        for text, chain in self.radiative.items():
            used = []
            onia_radiative.branching_ratio(chain, decay_block, self.default_decays, used,
                                           self.radiative_lo_widths)
            for what, value in used:
                logger.info('MadSpin: no BR line for %s in the param_card: %g, the PDG 2026 value'
                            % (what, value))
            logger.info('MadSpin: %s, a chain (%s), branching ratio %.4g', text,
                        ' then '.join(s.kind for s in chain.steps()),
                        onia_radiative.branching_ratio(chain, decay_block, self.default_decays,
                                                       lo_widths=self.radiative_lo_widths))

    def radiative_events(self, ms, pdg, nb_event, cumul, output_width, run_name, restrict_file):
        """MadSpin's generate_events for a bound state decaying by radiative
        chains: the decay events written by onia_radiative.PoolGenerator, one
        file per decay line (or one for all of them, cumul, mixed by their
        branching ratios); the partial widths are BR x total width."""
        import os
        import random
        import madgraph.various.lhe_parser as lhe_parser
        import MadSpin.onia_radiative as onia_radiative
        branches = self.radiative_branches(ms, pdg)
        decay_block = ms.banner.param_card['decay']
        totwidth = ms.banner.get('param_card', 'decay', abs(pdg)).value
        mass = ms.banner.get('param_card', 'mass', abs(pdg)).value
        model = ms._bookkeeping_model()

        def particle_mass(q):
            try:
                return ms.banner.get('param_card', 'mass', abs(q)).value
            except Exception:
                return 0.

        def colour_of(q):
            part = model.get_particle(q)
            if part is None:
                return 1
            c = part.get('color')
            return c if part.get('is_part') or c not in (3, -3) else -c

        rng = random.Random((int(ms.seed) if getattr(ms, 'seed', None) else 0) + 7 * abs(pdg))
        masses = dict(self.radiative_masses)
        out, channel_widths, width = {}, {}, (0. if cumul else 1.)
        chosen = [(i, text, self.radiative_chain_for_event(chain, pdg, ms))
                  for i, text, chain in branches if not restrict_file or i in restrict_file]
        for i, text, chain in chosen:
            for q in chain.final_pdgs():
                masses.setdefault(abs(q), particle_mass(q))
        gen = onia_radiative.PoolGenerator(self.radiative_evaluator, masses, rng, colour_of)
        brs = [onia_radiative.branching_ratio(chain, decay_block, self.default_decays,
                                              lo_widths=getattr(self, 'radiative_lo_widths', {}))
               for i, text, chain in chosen]
        groups = [chosen] if cumul else [[c] for c in chosen]
        for group in groups:
            gbrs = [brs[chosen.index(c)] for c in group]
            total = sum(gbrs)
            counts = [0] * len(group)
            for _ in range(int(nb_event)):
                x = rng.random() * total
                k = 0
                while k < len(group) - 1 and x > gbrs[k]:
                    x -= gbrs[k]
                    k += 1
                counts[k] += 1
            events = []
            for (i, text, chain), n in zip(group, counts):
                events += gen.generate(chain, n, mass)
            rng.shuffle(events)
            i0 = group[0][0]
            decay_dir = os.path.join(ms.path_me, 'decay_%s_%s' % (str(pdg).replace('-', 'x'), i0))
            # one file, outside Events/<run_name>: a refill (run_name
            # ms_refill_<gen>) then splits it into the per-worker slices
            # MadSpin's parallel unweighting reads (_generate_refill_pool)
            path = os.path.join(decay_dir, 'radiative_%s.lhe.gz' % run_name)
            gen.write(path, events, total * totwidth, mass)
            out[i0] = lhe_parser.EventFile(path)
            for (i, text, chain), br in zip(group, gbrs):
                channel_widths[i] = br * totwidth
            width = width + total * totwidth if cumul else width * total * totwidth
        if not output_width:
            return out
        return out, width, channel_widths

    def radiative_event_chains(self, ms):
        """Every radiative chain as its decay events carry it, for every bound
        state that decays by them (its colour octets and its antiparticle
        included). Several may share their legs (and so their tag)."""
        out = []
        for name, branches in ms.list_branches.items():
            fock = self.by_alias.get(name)
            if fock is None:
                continue
            pdgs = [fock.get('pdg_code')]
            anti = self.antiparticle(fock)
            if anti is not fock:
                pdgs.append(anti.get('pdg_code'))
            for text in branches:
                if text in self.radiative:
                    out += [self.radiative_chain_for_event(self.radiative[text], pdg, ms) for pdg in pdgs]
        return out

    def radiative_tags(self, ms):
        """{tag: all_me entry} of the decay events of the radiative chains,
        for every bound state that decays by them (its colour octets and its
        antiparticle included)."""
        out = {}
        for name, branches in ms.list_branches.items():
            fock = self.by_alias.get(name)
            if fock is None:
                continue
            pdgs = [fock.get('pdg_code')]
            anti = self.antiparticle(fock)
            if anti is not fock:
                pdgs.append(anti.get('pdg_code'))
            for text in branches:
                chain = self.radiative.get(text)
                if chain is None:
                    continue
                for pdg in pdgs:
                    c = self.radiative_chain_for_event(chain, pdg, ms)
                    out[c.tag()] = {'pdir': 'radiative_%s' % str(pdg).replace('-', 'x'),
                                    'order': ((c.parent_pdg,), tuple(c.final_pdgs())),
                                    'type': 'decay'}
        return out

    def wrap_decay_library(self, ms):
        """MadSpin's decay density library with the radiative chains added
        (onia_radiative.RadiativeLibrary) and their legs in pdg2prefix."""
        import MadSpin.onia_radiative as onia_radiative
        tags = self.radiative_tags(ms)
        if not tags:
            return
        base = ms.f2py_module[1] or None
        if isinstance(base, onia_radiative.RadiativeLibrary):
            return
        ms.f2py_module[1] = onia_radiative.RadiativeLibrary(
            base, self.radiative_evaluator, self.radiative_event_chains(ms))
        if not isinstance(ms.pdg2prefix[1], dict):
            ms.pdg2prefix[1] = {}
        for entry in tags.values():
            order = entry['order']
            ms.pdg2prefix[1][tuple(order[0]) + tuple(order[1])] = ('radiative', -1)

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
        if branches and all(b in self.radiative for b in branches):
            import MadSpin.onia_radiative as onia_radiative
            return sum(onia_radiative.branching_ratio(self.radiative[b], ms.banner.param_card['decay'],
                                                      self.default_decays,
                                                      lo_widths=getattr(self, 'radiative_lo_widths', {}))
                       for b in branches)
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
