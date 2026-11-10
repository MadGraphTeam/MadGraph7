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
        return base_objects.Particle({
            'name': alias(fock.get('name')),
            'antiname': alias(anti.get('name')),
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
        table = ms.banner.param_card['decay'].decay_table.get(abs(pdg))
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
    def onia_masses(onia_card_text):
        """{pdg: onium_mass} of an onia card (Block onium_mass)."""
        out = {}
        block = re.search(r'(?ims)^block\s+onium_mass\b(.*?)(?=^block|\Z)', onia_card_text or '')
        if block:
            for line in block.group(1).splitlines():
                line = line.split('#')[0].split()
                if len(line) >= 2:
                    out[int(line[0])] = float(line[1].replace('d', 'e').replace('D', 'E'))
        return out
