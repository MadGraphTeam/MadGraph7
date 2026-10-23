################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
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
"""Choose the default of the run-card parameter real_channels_x0 of a process,
i.e. whether the real-emission integration channels (section 6.3 of
arXiv:0908.4272) are worth enabling.

The Born channels generate a real-emission point from an underlying Born
point and the FKS radiation variables of one sector (i,j).  A propagator of
a real-emission diagram follows this parametrisation when its momentum does
not separate i from j: its virtuality is then a Born invariant (with i and j
merged), and the propagators that separate them only become singular when i
is soft or collinear to j, which the radiation variables sample.  A real
diagram is 'covered' if, for at least one of the FKS sectors of its real
process, none of its peaking propagators separates i from j.  A diagram that
is not covered has a structure that no Born channel follows: a t-channel
exchange or a resonance opened by the extra parton, as in u a > w+ h d or
u g > w+ d b b~ (gluon fusion into b b~).  Only then can the dedicated
real-emission channels improve the integration.
"""

import logging

logger = logging.getLogger('madgraph.fks_real_channels_auto')

# value of real_channels_x0 used when the channels are expected to help:
# a few percent gives the best performance in the benchmarks of the
# MadGraph7 paper (p p > w+ h [QED] and p p > w+ b b~ [QCD], 4FS)
AUTO_X0 = 0.05
# above this number of hard channels, the grid-adaptation overhead of the
# additional jobs is expected to outweigh the gain
MAX_HARD_CHANNELS = 1000


def diagram_propagators(diagram, legs):
    """Return, for each propagator of a base_objects.Diagram, the pair
    (set of external leg numbers on one side, pdg, s_channel).
    legs are the external legs of the process."""

    content = dict((l['number'], frozenset([l['number']])) for l in legs)
    props = []
    # the last vertex has no propagator; each other vertex produces one
    # leg, built from its other legs. Its 'state' is False for t-channel
    for vertex in diagram['vertices'][:-1]:
        side = frozenset()
        for leg in vertex['legs'][:-1]:
            side |= content[leg['number']]
        out = vertex['legs'][-1]
        content[out['number']] = side
        props.append((side, out['id'], out['state']))
    return props


class RealChannelAnalyser(object):
    """Classify the real-emission diagrams of FKS matrix elements"""

    def __init__(self, model, parameters):
        self.model = model
        self.parameters = parameters

    def value(self, name):
        """abs of a mass/width, None if unknown"""
        if name.upper() == 'ZERO':
            return 0.
        if self.parameters is None or name not in self.parameters:
            return None
        return abs(self.parameters[name].real)

    def mass(self, pdg):
        part = self.model.get_particle(pdg)
        if not part:
            return None
        return self.value(part.get('mass'))

    def peaking(self, side, pdg, s_channel, legs, initial, final_masses):
        """True if the propagator can enhance the integrand away from the
        soft and collinear limits"""

        part = self.model.get_particle(pdg)
        if not part:
            # fake propagator of a multi-leg vertex
            return False
        allnum = set(final_masses) | initial
        other = frozenset(allnum - side)
        # the s-channel propagator carrying all of sqrt(s-hat): no peak
        if side == initial or other == initial:
            return False
        mass = self.value(part.get('mass'))
        # final-state legs on the side of the propagator without the
        # initial state (with one initial leg for a t-channel propagator)
        finals = side if not (side & initial) else other - initial
        side_masses = [final_masses[n] for n in finals]
        if mass is None or None in side_masses:
            return True
        if mass == 0.:
            # a true (collinear-type) peak only if nothing on its side is
            # massive; otherwise the virtuality is bounded by these masses
            # and the propagator is as smooth as a massive one
            return sum(side_masses) == 0.
        if s_channel:
            width = self.value(part.get('width'))
            if width == 0.:
                return False
            # resonant only if heavier than the particles it decays into
            return mass > sum(side_masses) * (1. + 1e-10)
        # t-channel: enhanced when light compared to the hard scale,
        # bounded below by the final-state masses (without any final-state
        # mass, s-hat is set by the cuts, which exceed the EW masses)
        threshold = sum(final_masses.values())
        return threshold == 0. or mass < 0.5 * threshold

    def peak_propagators(self, amplitude):
        """For each diagram of amplitude (a real-emission base amplitude),
        its peaking propagators as [legs on one side, |pdg|]"""

        legs = amplitude['process']['legs']
        initial = frozenset(l['number'] for l in legs if not l['state'])
        final_masses = dict((l['number'], self.mass(l['id']))
                            for l in legs if l['state'])
        return [[[sorted(side), abs(pdg)] for side, pdg, s_channel in
                 diagram_propagators(diagram, legs)
                 if self.peaking(side, pdg, s_channel, legs, initial,
                                 final_masses)]
                for diagram in amplitude['diagrams']]


def canonical(side, nexternal):
    """The side of a propagator that does not contain leg 1"""
    side = frozenset(side)
    if 1 in side:
        side = frozenset(range(1, nexternal + 1)) - side
    return side


def leg_map(real_legs, born_legs, i, j, ij):
    """Born leg number of each real-emission leg but i: j becomes ij, and
    the other legs are matched by pdg and state (identical particles are
    interchangeable, since the Born diagrams are symmetric under their
    exchange)"""

    free = [(n, pdg, state) for n, pdg, state in born_legs if n != ij]
    mapping = {j: ij}
    for n, pdg, state in real_legs:
        if n in (i, j):
            continue
        for k, born in enumerate(free):
            if born[1:] == (pdg, state):
                mapping[n] = born[0]
                del free[k]
                break
        else:
            return None
    return mapping


class Sector(object):
    """An FKS sector (i, j) of a real process, with what the Born channels
    of its directory follow"""

    def __init__(self, i, j, soft, mapping, born_props, born_nexternal):
        self.i, self.j, self.soft = i, j, soft
        self.mapping = mapping
        self.born_props = born_props
        self.born_nexternal = born_nexternal

    def follows(self, side, apdg, nexternal):
        """True if the Born channels of this sector follow the propagator"""

        i, j = self.i, self.j
        if (i in side) == (j in side):
            return True
        # i attached to an external leg k: enhanced only when i is soft
        # (sampled by xi) or collinear to k (its own sector)
        if self.soft and (len(side) == 2 and i in side or
                          len(side) == nexternal - 2 and i not in side):
            return True
        # the same propagator, with i merged into j, in a Born diagram:
        # followed, up to the recoil of the emission
        if self.mapping is None:
            return False
        born_side = canonical([self.mapping[k] for k in side if k != i],
                              self.born_nexternal)
        return (born_side, apdg) in self.born_props


def covered(peaks, sectors, nexternal):
    """True if the Born channels of one of the sectors follow all the
    peaking propagators of a diagram"""

    return any(all(sector.follows(set(side), apdg, nexternal)
                   for side, apdg in peaks) for sector in sectors)


def identical_permutations(sectors, legs):
    """Close the list of (i, j, ...) under the exchange of identical
    legs: FKS keeps one sector per set of equivalent ones"""

    groups = {}
    for n, pdg, state in legs:
        groups.setdefault((pdg, state), []).append(n)
    swaps = [(a, b) for group in groups.values() if len(group) > 1
             for a in group for b in group if a < b]
    result = set(sectors)
    todo = list(result)
    while todo:
        sector = todo.pop()
        for a, b in swaps:
            swap = {a: b, b: a}
            new = (swap.get(sector[0], sector[0]),
                   swap.get(sector[1], sector[1])) + tuple(sector[2:])
            if new not in result:
                result.add(new)
                todo.append(new)
    return sorted(result)


_analysers = {}

def get_analyser(model):
    """One analyser (with the parameter values) per model"""
    if id(model) not in _analysers:
        from madgraph.interface.amcatnlo_interface import aMCatNLOInterface
        _analysers[id(model)] = RealChannelAnalyser(
            model, aMCatNLOInterface.get_parameter_values(model))
    return _analysers[id(model)]


def leg_list(process):
    return [[l['number'], l['id'], bool(l['state'])] for l in process['legs']]


def born_structure(born_amplitude):
    """Legs and propagators of the Born diagrams of a directory"""
    legs = born_amplitude['process']['legs']
    props = set()
    for diagram in born_amplitude['diagrams']:
        for side, pdg, s_channel in diagram_propagators(diagram, legs):
            props.add((tuple(sorted(canonical(side, len(legs)))), abs(pdg)))
    return {'legs': leg_list(born_amplitude['process']),
            'props': sorted(props)}


def real_structures(fks_matrix_element):
    """The information needed by decide() for one Born directory (an
    FKSHelasProcess), as JSON-friendly data"""

    born = fks_matrix_element.born_me.get('base_amplitude')
    reals = []
    for real in fks_matrix_element.real_processes:
        amp = real.matrix_element.get('base_amplitude')
        analyser = get_analyser(amp['process']['model'])
        reals.append({
            'legs': leg_list(amp['process']),
            'sectors': sorted(set((info['i'], info['j'],
                bool(info.get('need_color_links') or
                     info.get('need_charge_links')), info['ij'])
                for info in real.fks_infos)),
            'peaks': analyser.peak_propagators(amp)})
    return {'born': born_structure(born), 'reals': reals}


def decide(directories, n_hard_channels):
    """Combine the structures (real_structures) of all the Born
    directories. The sectors of a real process are collected over all the
    directories it belongs to, since the sector functions are normalised
    over all of them.
    Return (default of real_channels_x0, summary)"""

    reals = {}
    for directory in directories:
        born_legs = [tuple(l) for l in directory['born']['legs']]
        born_props = set((frozenset(side), apdg)
                         for side, apdg in directory['born']['props'])
        for real in directory['reals']:
            legs = [tuple(l) for l in real['legs']]
            # the same real process can have its final-state legs in a
            # different order in another directory: relabel them
            order = sorted(legs, key=lambda l: (l[2], l[2] and l[1], l[0]))
            relabel = dict((l[0], n + 1) for n, l in enumerate(order))
            legs = [(relabel[n], pdg, state) for n, pdg, state in order]
            key = tuple(l[1:] for l in legs)
            if key not in reals:
                reals[key] = {
                    'peaks': [[[sorted(relabel[k] for k in side), apdg]
                               for side, apdg in diagram]
                              for diagram in real['peaks']],
                    'nexternal': len(legs), 'sectors': []}
            sectors = [(relabel[i], relabel[j], soft, ij)
                       for i, j, soft, ij in real['sectors']]
            for i, j, soft, ij in identical_permutations(sectors, legs):
                mapping = leg_map(legs, born_legs, i, j, ij)
                reals[key]['sectors'].append(Sector(
                    i, j, soft, mapping, born_props, len(born_legs)))
    summary = {'n_reals': len(reals), 'n_diagrams': 0, 'n_uncovered': 0,
               'reals_uncovered': [], 'n_hard_channels': n_hard_channels}
    for key, entry in reals.items():
        uncovered = [n + 1 for n, peaks in enumerate(entry['peaks'])
                     if not covered(peaks, entry['sectors'],
                                    entry['nexternal'])]
        summary['n_diagrams'] += len(entry['peaks'])
        summary['n_uncovered'] += len(uncovered)
        if uncovered:
            summary['reals_uncovered'].append(
                ([pdg for pdg, state in key], uncovered, len(entry['peaks'])))
    if not summary['n_uncovered'] or n_hard_channels > MAX_HARD_CHANNELS:
        return -1., summary
    return AUTO_X0, summary
